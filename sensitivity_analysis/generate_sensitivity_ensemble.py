import subprocess
from datetime import datetime

import numpy as np

import sys
import os

cluster_login = 'robinmid'
subprocess.call(["bash", "/home/robin/scripts/mount_cluster_dirs.sh", "mount"])
sys.path.append("/home/robin/repos/acclimate-system-response/scripts")
sys.path.append("/home/robin/repos/post-processing/")

standard_parameters = {
    'possible_overcapacity_ratio': 1.15,
    'initial_storage_fill_factor': 10,
    'price_increase_production_extension': 5.0,
    'target_storage_refill_time': 2,
}

reference_run_unscaled = "/home/robinmid/repos/harvey_scaling/forcing/forcing_output/HARVEY_econYear2015_dT_0_3.2_0.2__re0_40000.0_2500.0__old_acclimate__ccFactor1.19/HARVEY_dT0.00_re0.nc"
reference_run_scaled = "/home/robinmid/repos/harvey_scaling/forcing/forcing_output/HARVEY_econYear2015_dT_0_3.2_0.2__re0_40000.0_2500.0__old_acclimate__ccFactor1.19/HARVEY_dT3.20_re40000.nc"
ensemble_dir = "/p/tmp/robinmid/acclimate_run/harvey_paper/sensitivity_analysis"
settings_template_path = "/home/robin/repos/harvey_scaling/sensitivity_analysis/settings_template.yml"
slurm_script_template = "/home/robin/repos/harvey_scaling/sensitivity_analysis/slurm_script_template.sh"
temp_dir = "/home/robin/repos/harvey_scaling/sensitivity_analysis/"

ensemble_parameters = {
    'initial_storage_fill_factor': np.arange(1, 31, 1),
    'price_increase_production_extension': np.arange(.5, 15.5, 0.5),
    'target_storage_refill_time': np.arange(0.5, 15.5, 0.5),
}


def generate_ensemble(parameters: dict, forcing_path, name=None, qos='medium', partition='standard', num_cpu=10,
                      start_runs=False):
    def exec_cluster(command):
        subprocess.call(["ssh", "{}@cluster.pik-potsdam.de".format(cluster_login), command])
    timestamp = str(datetime.now()).replace(' ', '_').split('.')[0]
    ensemble_directory = os.path.join(ensemble_dir, timestamp + (name if name is not None else ''))
    exec_cluster("mkdir {}".format(ensemble_directory))
    for parameter, parameter_values in parameters.items():
        parameter_dir = os.path.join(ensemble_directory, parameter)
        exec_cluster("mkdir {}".format(parameter_dir))
        for parameter_value in parameter_values:
            run_dir = os.path.join(parameter_dir, str(parameter_value))
            exec_cluster("mkdir {}".format(run_dir))
            with open(settings_template_path, "rt") as settings_file:
                settings_data = settings_file.read()
            for _param, _val in standard_parameters.items():
                if _param != parameter:
                    settings_data = settings_data.replace('+{}+'.format(_param), str(_val))
                else:
                    settings_data = settings_data.replace('+{}+'.format(_param), str(parameter_value))
            settings_data = settings_data.replace('+{}+'.format(parameter), parameter)
            settings_data = settings_data.replace('+forcing_filepath+', forcing_path)
            tempfile = os.path.join(temp_dir, 'settings.yml')
            with open(tempfile, "wt") as settings_file:
                settings_file.write(settings_data)
            subprocess.call(["scp", tempfile, "robinmid@cluster.pik-potsdam.de:{}/{}".format(run_dir, "settings.yml")])
            subprocess.call(["rm", tempfile])
            with open(slurm_script_template, "rt") as slurm_file:
                slurm_data = slurm_file.read()
            slurm_data = slurm_data.replace('+++qos+++', qos)
            slurm_data = slurm_data.replace('+++partition+++', partition)
            slurm_data = slurm_data.replace('+++workdir+++', run_dir)
            slurm_data = slurm_data.replace('+++num_cpu+++', str(num_cpu))
            tempfile = os.path.join(temp_dir, 'slurm_script.sh')
            with open(tempfile, "wt") as slurm_file:
                slurm_file.write(slurm_data)
            subprocess.call(["scp", tempfile, "robinmid@cluster.pik-potsdam.de:{}/{}".format(run_dir, "slurm_script.sh")])
            subprocess.call(["rm", tempfile])
            if start_runs:
                exec_cluster("sbatch {}/slurm_script.sh".format(run_dir))
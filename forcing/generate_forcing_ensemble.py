from __future__ import division
import warnings

warnings.filterwarnings('ignore')
import os
import argparse
import sys
import yaml
import pickle
import json
from netCDF4 import Dataset
import numpy as np
from os import path

home_dir = os.getenv("HOME")
sys.path.append(os.path.join(home_dir, 'repos/postproc/lib/'))
sys.path.append(os.path.join(home_dir, 'repos/acclimate/postproc/lib/'))
from acclimate import netcdf


def write_ncdf_output(_forcing_curves, _sector_list, _out_dir, _out_name, max_len=1000):
    max_len = max(max([len(curve) for curve in _forcing_curves.values()]), max_len)
    with Dataset(path.join(_out_dir, _out_name + '.nc'), 'w') as outfile:
        timedim = outfile.createDimension("time")
        timevar = outfile.createVariable("time", "f", "time")
        timevar[:] = np.arange(0, max_len)
        timevar.units = "days since 2009-01-01"
        timevar.calendar = "standard"

        regions = list(_forcing_curves.keys())
        regiondim = outfile.createDimension("region", len(regions))
        regionvar = outfile.createVariable("region", str, "region")
        for i, r in enumerate(regions):
            regionvar[i] = r

        sectors = _sector_list
        sectordim = outfile.createDimension("sector", len(sectors))
        sectorvar = outfile.createVariable("sector", str, "sector")
        for i, s in enumerate(sectors):
            sectorvar[i] = s

        forcing = outfile.createVariable("forcing", "f", ("time", "sector", "region"), zlib=True, complevel=7,
                                         fill_value=0)
        for reg, forcing_ts in _forcing_curves.items():
            for sec in sectors:
                forcing[:, sectors.index(sec), regions.index(reg)] = forcing_ts


# scale f0 with geographic extent
# scale tau with precipitation
def get_forcing_curve(_t0=0, _state='TX', _t_r_bi_initial=60, _f_r_bi=0.001, _t_max=1000, _radius_extension=0,
                      _temperature_change=0):
    forcing_params_path = os.path.join(home_dir, "repos/harvey_scaling/data/generated/initial_forcing_params.json")
    f0_m = json.load(open(forcing_params_path, 'rb'))['params'][_state]['m']
    f0_c = json.load(open(forcing_params_path, 'rb'))['params'][_state]['c']
    f0 = f0_c + f0_m * _radius_extension
    days = np.arange(_t_max - _t0)
    tau_bi = (-_t_r_bi_initial / np.log(_f_r_bi / f0_c)) * (1.07 ** _temperature_change)
    f_bi = f0 * np.exp(-days / tau_bi)
    f_bi[f_bi < _f_r_bi] = 0
    f_bi = np.concatenate((np.zeros(_t0), f_bi))
    return 1 - f_bi, f0, tau_bi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('econ_baseyear', type=int, help='')
    parser.add_argument('--ensemble_dir', type=str,
                        default=os.path.join(home_dir,
                                             'repos/harvey_scaling/data/forcing/ensembles'),
                        help='')
    parser.add_argument('--impact_time', type=int, default=5, help='')
    parser.add_argument('--sim_duration', type=int, default=600, help='')
    parser.add_argument('--min_dT', type=float, default=-0.5, help='')
    parser.add_argument('--max_dT', type=float, default=5.0, help='')
    parser.add_argument('--min_re', type=float, default=-10e3, help='')
    parser.add_argument('--max_re', type=float, default=100e3, help='')
    parser.add_argument('--dT_stepwidth', type=float, default=0.25, help='')
    parser.add_argument('--re_stepwidth', type=float, default=5e3, help='')
    pars = vars(parser.parse_args())

    settings_tpl = yaml.load(
        open(os.path.join(home_dir, 'repos/harvey_scaling/forcing/settings_template.yml'),
             'rb'))

    ensemble_dir_path = os.path.join(pars['ensemble_dir'],
                                     'HARVEY_econYear{}_noSec_expRecovery'.format(pars['econ_baseyear']))
    ensemble_dir_path = ensemble_dir_path + "_dT_{}_{}_{}__re{}_{}_{}".format(pars['min_dT'],
                                                                              pars['max_dT'],
                                                                              pars['dT_stepwidth'],
                                                                              pars['min_re'],
                                                                              pars['max_re'],
                                                                              pars['re_stepwidth'])
    if not os.path.exists(ensemble_dir_path):
        os.makedirs(ensemble_dir_path)
    ensemble_meta = {'scaled_scenarios': {}, 'base_scenario': {}, 'max_scenario': {}}
    dt_axis = np.arange(pars['min_dT'], pars['max_dT'] + pars['dT_stepwidth'], pars['dT_stepwidth'])
    re_axis = np.arange(pars['min_re'], pars['max_re'] + pars['re_stepwidth'], pars['re_stepwidth'])
    baseline = netcdf.load(
        os.path.join(home_dir, 'repos/harvey_scaling/data/external/EORA_{}_baseline.nc'.format(pars['econ_baseyear'])))
    baseline_production = baseline.agents.production[0, :, :]
    baseline_production *= 1e3
    sector_list = list(baseline.sector)
    for dt in dt_axis:
        for re in re_axis:
            f = {}
            for state in ['LA', 'TX']:
                eora_state = 'US.' + state
                X_baseline = baseline_production[:, eora_state].sum()
                f[state], f0, tau = get_forcing_curve(_state=state,
                                                      _t0=pars['impact_time'],
                                                      _t_max=pars['sim_duration'],
                                                      _radius_extension=re,
                                                      _temperature_change=dt
                                                      )
            iter_name = 'HARVEY_dT{1:.2f}_re{2:.0f}'.format(pars['econ_baseyear'], dt, re)
            write_ncdf_output(f, sector_list, ensemble_dir_path, iter_name)
            iter_scenario = {'scenario': {}, 'iter_name': '', 'params': {'f0': f0, 'tau': tau}}
            iter_scenario['scenario']['type'] = 'event_series'
            iter_scenario['scenario']['forcing'] = {}
            iter_scenario['scenario']['forcing']['variable'] = 'forcing'
            iter_scenario['scenario']['forcing']['file'] = os.path.join(ensemble_dir_path, iter_name + '.nc')
            iter_settings = settings_tpl
            iter_settings['scenarios'][0] = iter_scenario['scenario']
            settings_tpl['run']['stop'] = pars['sim_duration']
            settings_tpl['outputs'][0]['total'] = pars['sim_duration']
            settings_tpl['network']['file'] = '/p/projects/acclimate/data/eora/EORA{}_CHN_USA.nc'.format(
                pars['econ_baseyear'])
            iter_scenario['iter_name'] = iter_name
            iter_scenario['sim_duration'] = pars['sim_duration']
            ensemble_meta['scaled_scenarios'][(re, dt)] = iter_scenario
            yaml.dump(settings_tpl,
                      open(os.path.join(ensemble_dir_path, 'settings_{}.yml'.format(iter_name)), 'w'))
    with open(os.path.join(ensemble_dir_path, "ensemble_meta.pk"), 'wb') as meta_file:
        pickle.dump(ensemble_meta, meta_file)

from __future__ import division

import itertools
import warnings

import tqdm

warnings.filterwarnings('ignore')
import os
import argparse
import sys
import yaml
import pickle
import json
from netCDF4 import Dataset
import numpy as np

home_dir = os.getenv("HOME")
sys.path.append(os.path.join(home_dir, 'repos/postproc/lib/'))
sys.path.append(os.path.join(home_dir, 'repos/acclimate/postproc/lib/'))

all_sectors = [
    'AGRI', 'FISH', 'MINQ', 'FOOD', 'TEXL', 'WOOD', 'OILC', 'METL', 'MACH', 'TREQ', 'MANU', 'RECY', 'ELWA', 'CONS',
    'REPA', 'WHOT', 'RETT', 'GAST', 'TRAN', 'COMM', 'FINC', 'ADMI', 'EDHE', 'HOUS', 'OTHE', 'REXI',
    # 'FCON'
]


def write_ncdf_output(_forcing_curves, _sector_list, _out_dir, _out_name, max_len=1000):
    max_len = max(max([len(curve) for curve in _forcing_curves.values()]), max_len)
    with Dataset(os.path.join(_out_dir, _out_name + '.nc'), 'w') as outfile:
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
def get_forcing_curves(_t0=0, _cc_factor=1.19, _t_r_init=60, _f_r=0.001, _t_max=1000, _re=0, _dT=0):
    forcing_params_path = os.path.join(home_dir, "repos/harvey_scaling/data/generated/initial_forcing_params.json")
    forcing_curves = {}
    forcing_params = {}
    t_r = _t_r_init * (_cc_factor ** _dT)
    days = np.arange(_t_max - _t0)
    for state in ['TX', 'LA']:
        f0_m = json.load(open(forcing_params_path, 'rb'))['params'][state]['m']
        f0_c = json.load(open(forcing_params_path, 'rb'))['params'][state]['c']
        f0 = f0_c + f0_m * _re
        tau = -t_r / np.log(_f_r / f0)
        f = f0 * np.exp(-days / tau)
        t_r = int(np.ceil(t_r))
        f[t_r + 1:] = 0
        f = np.concatenate((np.zeros(_t0), f))
        forcing_curves['US.' + state] = 1 - f
        forcing_params['US.' + state] = {'f_0': f0, 'tau': tau}
    forcing_params['all'] = {'t_r': t_r, 'f_r': _f_r}
    return forcing_curves, forcing_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('econ_baseyear', type=int, help='')
    parser.add_argument('--ensemble_dir', type=str,
                        default=os.path.join(home_dir, 'repos/harvey_scaling/forcing/forcing_output'), help='')
    parser.add_argument('--impact_time', type=int, default=5, help='')
    parser.add_argument('--sim_duration', type=int, default=380, help='')
    parser.add_argument('--min_dT', type=float, default=0, help='')
    parser.add_argument('--max_dT', type=float, default=3.2, help='')
    parser.add_argument('--min_re', type=float, default=0, help='')
    parser.add_argument('--max_re', type=float, default=40e3, help='')
    parser.add_argument('--dT_stepwidth', type=float, default=0.2, help='')
    parser.add_argument('--re_stepwidth', type=float, default=2.5e3, help='')
    parser.add_argument('--cc_factor', type=float, default=1.197, help='')
    parser.add_argument('--slopes', type=str, default='', help='in km/degC SST, not GMT!!') # in km/degC SST, not GMT!!
    parser.add_argument('--binstep_re', type=float, default=1e3, help='')
    parser.add_argument('--binstep_dT', type=float, default=0.08, help='')
    parser.add_argument('--binstep_nr', type=float, default=1, help='')
    parser.add_argument('--possible_overcapacity', type=float, default=1.15, help='')
    parser.add_argument('--old_acclimate', action='store_true')
    pars = vars(parser.parse_args())

    if not pars['old_acclimate']:
        settings_tpl_path = os.path.join(home_dir, 'repos/harvey_scaling/forcing/settings_template.yml')
    elif pars['old_acclimate']:
        settings_tpl_path = os.path.join(home_dir, 'repos/harvey_scaling/forcing/settings_template_old_acclimate_version.yml')

    ensemble_dir_path = os.path.join(pars['ensemble_dir'], 'HARVEY_econYear{}'.format(pars['econ_baseyear']))
    ensemble_dir_path += "_dT_{}_{}_{}".format(pars['min_dT'], pars['max_dT'], pars['dT_stepwidth'])
    if pars['slopes'] != '':
        ensemble_dir_path += "__slopes{}".format(pars['slopes'])
    else:
        ensemble_dir_path += "__re{}_{}_{}".format(pars['min_re'], pars['max_re'], pars['re_stepwidth'])
    if pars['possible_overcapacity'] != 1.15:
        ensemble_dir_path += "__overcapa_{}".format(pars['possible_overcapacity'])
    if pars['old_acclimate']:
        ensemble_dir_path += "__old_acclimate"
    ensemble_dir_path += "__ccFactor{}".format(pars['cc_factor'])

    ensemble_meta = {'scaled_scenarios': {}}

    if pars['slopes'] == '':
        dt_axis = np.arange(pars['min_dT'], pars['max_dT'] + pars['dT_stepwidth'], pars['dT_stepwidth'])
        re_axis = np.arange(pars['min_re'], pars['max_re'] + pars['re_stepwidth'], pars['re_stepwidth'])
        dt_re_pairs = itertools.product(dt_axis, re_axis)
    else:
        ensemble_meta['slopes'] = {}
        slopes = [int(i) for i in pars['slopes'].split('+')]
        dt_re_pairs = []
        for slope in slopes:
            ensemble_meta['slopes'][slope] = {}
            for dt in np.arange(pars['min_dT'], pars['max_dT'] + pars['dT_stepwidth'], pars['dT_stepwidth']):
                re = slope * dt
                ensemble_meta['slopes'][slope][(dt, re)] = []
                for dt_ in np.arange(dt - pars['binstep_nr'] * pars['binstep_dT'], dt + pars['binstep_dT'] * (pars['binstep_nr'] + 0.5), pars['binstep_dT']):
                    for re_ in np.arange(re - pars['binstep_nr'] * pars['binstep_re'], re + pars['binstep_re'] * (pars['binstep_nr'] + 0.5), pars['binstep_re']):
                        dt_re_pairs.append((dt_, re_))
                        ensemble_meta['slopes'][slope][(dt, re)].append((dt_, re_))
        dt_re_pairs = list(set(dt_re_pairs))

    if not os.path.exists(ensemble_dir_path):
        os.makedirs(ensemble_dir_path)
    for dt, re in tqdm.tqdm(dt_re_pairs):
        forcing_curves, forcing_params = get_forcing_curves(_t0=pars['impact_time'], _cc_factor=pars['cc_factor'],
                                                            _t_max=pars['sim_duration'], _re=re, _dT=dt)
        iter_name = 'HARVEY_dT{1:.2f}_re{2:.0f}'.format(pars['econ_baseyear'], dt, re)
        write_ncdf_output(forcing_curves, all_sectors, ensemble_dir_path, iter_name)
        forcing_filepath = os.path.join(ensemble_dir_path, iter_name + '.nc')
        with open(settings_tpl_path, "rt") as settings_file:
            settings_data = settings_file.read()
        settings_data = settings_data.replace('+++stop_time+++', str(pars['sim_duration']))
        settings_data = settings_data.replace('+++forcing_filepath+++', forcing_filepath)
        settings_data = settings_data.replace('+++overcapacity_ratio+++', str(pars['possible_overcapacity']))
        with open(os.path.join(ensemble_dir_path, 'settings_{}.yml'.format(iter_name)), "wt") as settings_file:
            settings_file.write(settings_data)
        ensemble_meta['scaled_scenarios'][(int(re), float(dt))] = {
            'name': iter_name,
            'params': forcing_params,
            'forcing_file': forcing_filepath,
            'duration': pars['sim_duration'],
        }
    with open(os.path.join(ensemble_dir_path, "ensemble_meta.pk"), 'wb') as meta_file:
        pickle.dump(ensemble_meta, meta_file)

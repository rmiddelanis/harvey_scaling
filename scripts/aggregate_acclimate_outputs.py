import sys

sys.path.append("/home/robin/repos/post-processing")
sys.path.append("/home/robinmid/repos/post-processing")
from acclimate.dataset import AcclimateOutput

sys.path.append('/home/robin/repos/harvey_scaling/scripts')
sys.path.append('/home/robinmid/repos/harvey_scaling/scripts')
from dataformat import AggrData

import tqdm
import argparse
import os
import xarray as xr
import numpy as np
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('output_name', type=str)
    parser.add_argument('--no_legacy_output', action='store_true')
    parser.add_argument('--old_acclimate_format', action='store_true')
    parser.add_argument('--lazy_loading_off', action='store_true')
    parser.add_argument('--sectors_to_keep', type=str, default=None)
    parser.add_argument('--variables', type=str, default='production_quantity')
    pars = vars(parser.parse_args())
    input_dir = pars['input_dir']
    output_dir = pars['output_dir']
    output_name = pars['output_name']
    variables = pars['variables'].split('*')
    compound_variables = [v for v in variables if '-' in v or '+' in v]
    variables = [v for v in variables if v not in compound_variables]
    no_legacy_output = pars['no_legacy_output']
    old_acclimate_format = pars['old_acclimate_format']
    sectors_to_keep = pars['sectors_to_keep']
    if sectors_to_keep is not None:
        sectors_to_keep = sectors_to_keep.split('*')
    else:
        sectors_to_keep = ['ALL_INDUSTRY']
    res = None
    dT_list = []
    re_list = []
    for idx, run_dir in tqdm.tqdm(enumerate(os.listdir(input_dir))):
        output_file = os.path.join(input_dir, run_dir, 'output.nc')
        if os.path.exists(os.path.join(output_file)):
            dT = float(run_dir.split('_')[1][2:])
            re = int(run_dir.split('_')[2][2:])
            _data = AcclimateOutput(filename=output_file, groups_to_load=['firms'],
                                    old_output_format=old_acclimate_format, lazy_loading=(not pars['lazy_loading_off']))
            _data = _data.expand_dims(['temp'])
            _data = _data[variables]
            dT_list.append(dT)
            re_list.append(re)
            if res is None:
                res = _data
            else:
                res._data = xr.concat([res.data, _data.data], dim='temp')
    for cv in compound_variables:
        if '+' in cv:
            va, vb = cv.split('+')
            res.data[cv] = res.data[va] + res.data[vb]
        elif '-' in cv:
            va, vb = cv.split('-')
            res.data[cv] = res.data[va] - res.data[vb]
    res.data['dT'] = dT_list
    res.data['re'] = re_list
    res = res.set_index(temp=['dT', 're']).unstack(dim='temp')
    res.data.to_netcdf(os.path.join(output_dir, "{}.nc".format(output_name.replace('.nc', ''))))
    res.baseline.to_netcdf(os.path.join(output_dir, "{}_baseline.nc".format(output_name.replace('.nc', ''))))

    if not no_legacy_output:
        industry_sectors = list(set(res.agent_sector.values) - {'FCON'})
        res.group_agents(dim='sector', group=industry_sectors, name='ALL_INDUSTRY', inplace=True, drop=False)
        res = res.sel(agent_sector=sectors_to_keep)
        res = res.set_index(agent=['agent_sector', 'agent_region']).unstack('agent').drop_vars('agent_type')
        res = res.to_array(dim='variable')
        res = res.transpose('variable', 'agent_region', 'agent_sector', 're', 'dT', 'time')
        region_index = {c: [c] for c in res.agent_region.values}
        sector_index = {s: [s] for s in res.agent_sector.values}
        sector_index['ALL_INDUSTRY'] = industry_sectors
        old_data = AggrData(np.ma.masked_array(data=res.data.data, mask=np.isnan(res.data.data)),
                            res.coords['variable'].values, region_index, sector_index, res.coords['re'].values,
                            res.coords['dT'].values)
        pk_name = "{}__{}.pk".format(output_name.replace('.nc', ''), '_'.join(s for s in sectors_to_keep))
        outfile = open(os.path.join(output_dir, pk_name), 'wb')
        pickle.dump(old_data, outfile)
        print("finished writing file {}.".format(outfile))
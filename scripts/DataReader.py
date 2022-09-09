import numpy as np
from multiprocessing import sharedctypes, Pool, process
import itertools
import sys
from netCDF4 import Dataset
import os
import argparse
import timeit

sys.path.append('/home/robinmid/repos/harvey_scaling/')
sys.path.append('/home/robin/repos/harvey_scaling/')
from analysis.utils import WORLD_REGIONS, SECTOR_GROUPS, get_axes_and_dirs, pickle_data, \
    get_regions_dict, get_sectors_dict, make_figure_dir, get_index_list
from analysis.dataformat import AggrData

global data_array_shared
global mask_array_shared

process.current_process()._config['tempdir'] = '/p/tmp/robinmid/temp'


class DataReader:
    def __init__(self, _vars, _ensemble_meta, _experiment_series_dir, _region_groups: dict, _sector_groups: dict,
                 _lambda_axis, _duration_axis, _time_frame, _num_cpus=0):
        self.vars = _vars
        self.ensemble_meta = _ensemble_meta
        self.experiment_series_dir = _experiment_series_dir
        self.region_groups = _region_groups
        self.sector_groups = _sector_groups
        self.lambda_axis = _lambda_axis
        self.duration_axis = _duration_axis
        self.time_frame = _time_frame
        self.num_cpus = _num_cpus

        self.region_indices = {}
        self.sector_indices = {}

        data_array = np.zeros((len(self.vars), len(self.region_groups), len(self.sector_groups), len(self.lambda_axis),
                               len(self.duration_axis), self.time_frame))
        mask_array = data_array != 0

        data_array_ctypes = np.ctypeslib.as_ctypes(data_array)
        mask_array_ctypes = np.ctypeslib.as_ctypes(mask_array)

        global data_array_shared
        global mask_array_shared
        data_array_shared = sharedctypes.RawArray(data_array_ctypes._type_, data_array_ctypes)
        mask_array_shared = sharedctypes.RawArray(mask_array_ctypes._type_, mask_array_ctypes)

    def go(self):
        l_d_tuples = itertools.product(self.lambda_axis, self.duration_axis)
        if self.num_cpus == 0:
            self.num_cpus = os.cpu_count()
        print("Starting {} workers.".format(self.num_cpus))
        p = Pool(self.num_cpus)
        p.starmap(self.read_file, l_d_tuples)
        result_data = np.ctypeslib.as_array(data_array_shared)
        result_mask = np.ctypeslib.as_array(mask_array_shared)
        result = np.ma.array(result_data, mask=result_mask)

        return AggrData(result, np.array(self.vars), self.region_groups, self.sector_groups, np.array(self.lambda_axis),
                        np.array(self.duration_axis))

    def read_file(self, lambda_value_, duration_value_):
        l_ = np.where(self.lambda_axis == lambda_value_)[0]
        d_ = np.where(self.duration_axis == duration_value_)[0]
        if (lambda_value_, duration_value_) not in ensemble_meta['scaled_scenarios'].keys():
            print("##### Key ({}, {}) not found. Skipping.".format(lambda_value_, duration_value_))
            return
        iteration_path = os.path.join(self.experiment_series_dir,
                                      self.ensemble_meta['scaled_scenarios'][(lambda_value_, duration_value_)][
                                          'iter_name'] + "/output.nc")
        print("##### Reading from file {}".format(iteration_path))
        try:
            dataset = Dataset(iteration_path, "r", format="NETCDF4")
        except Exception as e:
            print("File {} could not be loaded.".format(iteration_path))
            print(e)
            return
        iteration_sim_duration = len(dataset.variables['time'])
        iteration_time_frame = self.time_frame
        if iteration_sim_duration < iteration_time_frame:
            print("Attention! Dataset {} contains less time steps ({}) than requested ({}). Continue with "
                  "dataset length instead, mask remaining length.".format(iteration_path, iteration_sim_duration,
                                                                          self.time_frame))
            iteration_time_frame = iteration_sim_duration
        data_tmp = np.ctypeslib.as_array(data_array_shared)
        mask_tmp = np.ctypeslib.as_array(mask_array_shared)
        for v, var in enumerate(self.vars):
            _data = dataset["/agents/" + var.replace('/', '')][:].data
            _data = np.ma.array(_data, mask=np.isnan(_data), fill_value=0)
            for r, region_group_name in enumerate(self.region_groups):
                if region_group_name in self.region_indices.keys():
                    iter_region_indices = self.region_indices[region_group_name]
                else:
                    iter_region_indices = get_index_list(self.region_groups[region_group_name], dataset["/region"][:])
                    self.region_indices[region_group_name] = iter_region_indices
                for s, sector_group_name in enumerate(self.sector_groups):
                    if sector_group_name in self.sector_indices.keys():
                        iter_sector_indices = self.sector_indices[sector_group_name]
                    else:
                        iter_sector_indices = get_index_list(self.sector_groups[sector_group_name],
                                                             dataset["/sector"][:])
                        self.sector_indices[sector_group_name] = iter_sector_indices
                    iter_series = np.sum(
                        np.sum(_data[:iteration_time_frame, iter_sector_indices, :][:, :, iter_region_indices], axis=1),
                        axis=1)
                    data_tmp[v, r, s, l_, d_, :iteration_time_frame] = iter_series
                    mask_tmp[v, r, s, l_, d_, iteration_time_frame:] = data_tmp[v, r, s, l_, d_,
                                                                       iteration_time_frame:] == 0
        dataset.close()
        del _data
        del data_tmp
        del mask_tmp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('ensemble_dir', type=str, help='')
    parser.add_argument('--num_cpus', type=int, default=0, help='')
    parser.add_argument('--variable', type=str, default='all', help='Variable to read. If not specified, standard '
                                                                    'set of variables will beused.')
    parser.add_argument('--region', type=str, default='all', help='Region to read. If not specified, standard set'
                                                                  'of regions will beused.')
    parser.add_argument('--sector', type=str, default='all', help='Sector to read. If not specified, standard set'
                                                                  'of sectors will beused.')
    parser.add_argument('--output', type=str, default='', help='Output file name.')
    parser.add_argument('--max_timesteps', type=int, default=0, help='')
    pars = vars(parser.parse_args())
    starting_time = timeit.default_timer()
    experiment_series_dir = pars['ensemble_dir']
    make_figure_dir(experiment_series_dir)
    lambda_axis, duration_axis, ensemble_meta = get_axes_and_dirs(experiment_series_dir)
    num_cpus = pars['num_cpus']
    time_frame = max(
        [simulation_meta['sim_duration'] for simulation_meta in list(ensemble_meta['scaled_scenarios'].values())])
    if pars['max_timesteps'] > 0:
        time_frame = min(time_frame, pars['max_timesteps'])
    if pars['variable'] == 'all':
        variables = ['consumption', 'consumption_price', 'consumption_value', 'demand', 'demand_price', 'demand_value',
                     'production', 'production_price', 'production_value', 'storage', 'storage_price', 'storage_value',
                     'total_loss', 'total_loss_price', 'total_loss_value', 'total_value_loss', 'offer_price',
                     'expected_offer_price', 'expected_production', 'expected_production_price',
                     'expected_production_value', 'communicated_possible_production',
                     'communicated_possible_production_price', 'communicated_possible_production_value',
                     'unit_production_costs', 'total_production_costs', 'total_revenue', 'direct_loss',
                     'direct_loss_price', 'direct_loss_value', 'forcing', 'incoming_demand', 'incoming_demand_price',
                     'incoming_demand_value', 'production_capacity', 'desired_production_capacity',
                     'possible_production_capacity']
    elif pars['variable'] == 'set_harvey':
        variables = ['production', 'production_value', 'incoming_demand', 'incoming_demand_value', 'consumption',
                     'consumption_value', 'forcing']
    else:
        variables = pars['variable'].split('+')
    if pars['region'] == 'all':
        regions = set(list(WORLD_REGIONS.keys()) + WORLD_REGIONS['WORLD'])
    elif pars['region'] == 'set_1':
        regions = set(
            list(WORLD_REGIONS.keys()) + ['DEU', 'USA', 'CHN', 'CAN', 'ESP', 'FRA', 'GBR', 'IND', 'JPN', 'MEX', 'POL',
                                          'HUN'])
    elif pars['region'] == 'set_sandy':
        regions = set(list(WORLD_REGIONS['USA']) + ['DEU', 'CHN', 'EUR', 'WORLD', 'ROW', 'USA_REST_SANDY'])
    else:
        regions = pars['region'].split('+')
    if pars['sector'] == 'all':
        sectors = [i for i in SECTOR_GROUPS['ALLSECTORS']] + ['PRIVSECTORS']
    else:
        sectors = pars['sector'].split('+')
    region_args = get_regions_dict(regions)
    sector_args = get_sectors_dict(sectors)
    data_reader = DataReader(variables, ensemble_meta, experiment_series_dir, region_args, sector_args, lambda_axis,
                             duration_axis, time_frame, num_cpus)
    data = data_reader.go()
    if pars['output'] == '':
        filename = "{}vars_{}regns_{}sects_{}lambda_{}duration".format(len(variables), len(region_args),
                                                                             len(sector_args), len(lambda_axis),
                                                                             len(duration_axis))
    else:
        filename = pars['output']
    pickle_data(data.data_capsule, 'data_cap', experiment_series_dir, filename)
    runtime = timeit.default_timer() - starting_time
    print("Runtime was {}".format(runtime))

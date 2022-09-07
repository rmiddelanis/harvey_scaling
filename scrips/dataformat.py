import copy

import numpy as np
import sys
import pandas as pd
import xarray as xr

sys.path.append('/home/robinmid/repos/hurricanes_hindcasting_remake/')
sys.path.append('/home/robin/repos/hurricanes_hindcasting_remake/')
from analysis.utils import get_index_list, detect_stationarity_and_offset_in_series, WORLD_REGIONS

sys.path.append('/home/robin/repos/post-processing//')
from acclimate.dataset import AcclimateOutput


class DataCapsule:
    def __init__(self, _data: np.ndarray, _variables: np.ndarray, _regions: dict, _sectors: dict,
                 _lambda_axis: np.ndarray, _duration_axis: np.ndarray):
        if type(_data) is not np.ma.core.MaskedArray:
            raise ValueError("_data must be of type np.ndarray.")
        if type(_variables) is not np.ndarray:
            raise ValueError("_variables must be of type np.ndarray.")
        if type(_lambda_axis) is not np.ndarray:
            raise ValueError("_lambda_axis must be of type np.ndarray.")
        if type(_duration_axis) is not np.ndarray:
            raise ValueError("_duration_axis must be of type np.ndarray.")
        self.duration_axis = _duration_axis
        self.lambda_axis = _lambda_axis
        self.sectors = _sectors
        self.regions = _regions
        self.variables = _variables
        self.data = _data
        self.shape = _data.shape


# noinspection DuplicatedCode
class AggrData:
    def __init__(self, *args, _base_damage=None, _base_forcing=None, _scaled_scenarios=None, _slope_meta=None):
        if len(args) == 1:
            if type(args[0]).__name__ == "DataCapsule":
                self.data_capsule = args[0]
            else:
                raise TypeError('Must pass argument of type DataCapsule or six arguments')
        elif len(args) == 6:
            self.data_capsule = DataCapsule(*args)
        else:
            raise TypeError('Must pass one argument of type DataCapsule or six arguments')
        self.base_damage = _base_damage
        self.base_forcing = _base_forcing
        self.scaled_scenarios = _scaled_scenarios
        self.slope_meta = _slope_meta

    def get_vars(self, _vars=None):
        if _vars is None:
            return self.data_capsule.variables
        else:
            if not isinstance(_vars, (list, tuple, np.ndarray)):
                _vars = [_vars]
            _vars = np.array(_vars)
            vars_indices = get_index_list(_vars, self.data_capsule.variables)
            data_new = self.data_capsule.data[vars_indices, ...]
            return AggrData(data_new, self.data_capsule.variables[vars_indices], self.data_capsule.regions,
                            self.data_capsule.sectors, self.data_capsule.lambda_axis,
                            self.data_capsule.duration_axis, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios,
                            _slope_meta=self.slope_meta)

    def get_regions(self, _regions=None):
        if _regions is None:
            return self.data_capsule.regions
        else:
            if not isinstance(_regions, (list, tuple, np.ndarray)):
                _regions = [_regions]
            _regions = np.array(_regions)
            regions_indices = get_index_list(_regions, list(self.data_capsule.regions.keys()))
            regions_new = {}
            for _region in _regions:
                if _region in self.data_capsule.regions.keys():
                    regions_new[_region] = self.data_capsule.regions[_region]
            data_new = self.data_capsule.data[:, regions_indices, ...]
            return AggrData(data_new, self.data_capsule.variables, regions_new, self.data_capsule.sectors,
                            self.data_capsule.lambda_axis, self.data_capsule.duration_axis,
                            _base_damage=self.base_damage, _base_forcing=self.base_forcing,
                            _scaled_scenarios=self.scaled_scenarios, _slope_meta=self.slope_meta)

    def get_sectors(self, _sectors=None):
        if _sectors is None:
            return self.data_capsule.sectors
        else:
            if not isinstance(_sectors, (list, tuple, np.ndarray)):
                _sectors = [_sectors]
            _sectors = np.array(_sectors)
            sectors_indices = get_index_list(_sectors, list(self.data_capsule.sectors.keys()))
            sectors_new = {}
            for _sector in _sectors:
                if _sector in self.data_capsule.sectors.keys():
                    sectors_new[_sector] = self.data_capsule.sectors[_sector]
            data_new = self.data_capsule.data[:, :, sectors_indices, ...]
            return AggrData(data_new, self.data_capsule.variables, self.data_capsule.regions, sectors_new,
                            self.data_capsule.lambda_axis, self.data_capsule.duration_axis,
                            _base_damage=self.base_damage, _base_forcing=self.base_forcing,
                            _scaled_scenarios=self.scaled_scenarios, _slope_meta=self.slope_meta)

    def get_re(self, _lambdavals=None):
        if _lambdavals is None:
            return self.data_capsule.lambda_axis
        else:
            if not isinstance(_lambdavals, (list, tuple, np.ndarray)):
                _lambdavals = [_lambdavals]
            _lambdavals = np.array(_lambdavals)
            lambda_indices = get_index_list(_lambdavals, self.data_capsule.lambda_axis)
            data_new = self.data_capsule.data[..., lambda_indices, :, :]
            return AggrData(data_new, self.data_capsule.variables, self.data_capsule.regions, self.data_capsule.sectors,
                            self.data_capsule.lambda_axis[lambda_indices],
                            self.data_capsule.duration_axis, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios,
                            _slope_meta=self.slope_meta)

    def get_dt(self, _durationvals=None):
        if _durationvals is None:
            return self.data_capsule.duration_axis
        else:
            if not isinstance(_durationvals, (list, tuple, np.ndarray)):
                _durationvals = [_durationvals]
            _durationvals = np.array(_durationvals)
            duration_indices = get_index_list(_durationvals, self.data_capsule.duration_axis)
            data_new = self.data_capsule.data[..., duration_indices, :]
            return AggrData(data_new, self.data_capsule.variables, self.data_capsule.regions, self.data_capsule.sectors,
                            self.data_capsule.lambda_axis,
                            self.data_capsule.duration_axis[duration_indices], _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios,
                            _slope_meta=self.slope_meta)

    def get_dt_axis(self):
        return self.data_capsule.duration_axis

    def get_re_axis(self):
        return self.data_capsule.lambda_axis

    def clip(self, *args):
        if len(args) == 1:
            _from = 0
            _to = args[0]
        elif len(args) > 1:
            _from = args[0]
            _to = args[1]
        else:
            raise ValueError('Must pass at least one argument _from or both _from and _to')
        return AggrData(self.data_capsule.data[..., _from:_to], self.data_capsule.variables, self.data_capsule.regions,
                        self.data_capsule.sectors, self.data_capsule.lambda_axis,
                        self.data_capsule.duration_axis, _base_damage=self.base_damage, _base_forcing=self.base_forcing,
                        _scaled_scenarios=self.scaled_scenarios, _slope_meta=self.slope_meta)

    @property
    def data(self):
        return self.get_data()

    def get_data(self):
        return self.data_capsule.data

    @property
    def shape(self):
        return self.data_capsule.data.shape

    @property
    def dT_stepwidth(self):
        return self.data_capsule.duration_axis[1] - self.data_capsule.duration_axis[0]

    @property
    def re_stepwidth(self):
        return self.data_capsule.lambda_axis[1] - self.data_capsule.lambda_axis[0]

    def get_sim_duration(self):
        return self.data_capsule.data.shape[-1]

    def add_var(self, _data, _name, _inplace=False):
        if _data.shape[1:-1] != self.get_data().shape[1:-1]:
            raise ValueError('Must pass array same dimensions in region, sector, lambda and duration!')
        if _data.shape[-1] < self.get_data().shape[-1]:
            raise UserWarning('Attention. New variable is shorter than existing data. Will mask remaining time steps.')
        new_dim = np.ma.masked_all((1,) + self.get_data().shape[1:])
        new_dim[..., :_data.shape[-1]] = _data
        data_new = np.ma.concatenate([self.get_data(), new_dim], axis=0)
        vars_new = np.concatenate([self.data_capsule.variables, np.array([_name])])
        if _inplace:
            self.data_capsule.data = data_new
            self.data_capsule.variables = np.concatenate([self.data_capsule.variables, np.array([_name])])
        else:
            return AggrData(data_new, vars_new, self.data_capsule.regions,
                            self.data_capsule.sectors, self.data_capsule.lambda_axis,
                            self.data_capsule.duration_axis, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios,
                            _slope_meta=self.slope_meta)

    def drop_var(self, _name, _inplace=False):
        if _name not in self.get_vars():
            print('Variable {} is not contained in data. Doing nothing.'.format(_name))
            return
        vars_new = self.get_vars()[self.get_vars() != _name]
        data_new = self.get_vars(vars_new).get_data()
        if _inplace:
            self.data_capsule.data = data_new
            self.data_capsule.variables = vars_new
        else:
            return AggrData(data_new, vars_new, self.data_capsule.regions,
                            self.data_capsule.sectors, self.data_capsule.lambda_axis,
                            self.data_capsule.duration_axis, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios,
                            _slope_meta=self.slope_meta)

    def drop_region(self, _name, _inplace=False):
        if _name not in self.get_regions():
            print('Region {} is not contained in data. Doing nothing.'.format(_name))
            return
        regions_new = copy.deepcopy(self.get_regions())
        regions_new.pop(_name)
        data_new = self.get_regions(list(regions_new.keys())).get_data()
        if _inplace:
            self.data_capsule.data = data_new
            self.data_capsule.regions = regions_new
        else:
            return AggrData(data_new, self.data_capsule.variables, regions_new,
                            self.data_capsule.sectors, self.data_capsule.lambda_axis,
                            self.data_capsule.duration_axis, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios,
                            _slope_meta=self.slope_meta)

    def calc_prices(self, vars=None, _inplace=False):
        if vars == None:
            vars = ['communicated_possible_production', 'consumption', 'demand', 'direct_loss',
                    'expected_production', 'incoming_demand', 'production', 'storage', 'total_loss']
        add_vars = []
        for var in vars:
            if var in self.get_vars() and var + "_value" in self.get_vars() and var + "_price" not in self.get_vars():
                add_vars.append(var)
        if _inplace:
            for var in add_vars:
                self.add_var(self.get_vars(var + "_value").get_data() / self.get_vars(var).get_data(), var + "_price",
                             _inplace=True)
        else:
            data_new = np.ma.masked_all((len(add_vars),) + self.shape[1:])
            vars_new = self.get_vars()
            for var_idx, var in enumerate(add_vars):
                data_new[var_idx, ...] = self.get_vars(var + "_value").get_data() / self.get_vars(var).get_data()
                vars_new = np.concatenate([vars_new, np.array([var + "_value"])])
            data_new = np.ma.concatenate([self.get_data(), data_new], axis=0)
            return AggrData(data_new, vars_new, self.data_capsule.regions,
                            self.data_capsule.sectors, self.data_capsule.lambda_axis,
                            self.data_capsule.duration_axis, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios,
                            _slope_meta=self.slope_meta)

    def calc_eff_prod_capacity(self, _inplace=False):
        if 'effective_production_capacity' in self.get_vars():
            print('Variable \'effective_production_capacity\' already in data. Doing nothing.')
            return
        if 'production' not in self.get_vars() or 'effective_forcing' not in self.get_vars():
            print(
                'Variables \'production\' and \'effective_forcing\' needed to calculate production_capacity. Doing nothing.')
            return
        eff_prod_capacity = self.get_vars('production').get_data() / (
                self.get_vars('effective_forcing').get_data() * self.get_vars('production').clip(1).get_data())
        if _inplace:
            self.add_var(eff_prod_capacity, 'effective_production_capacity', _inplace=True)
        else:
            data_new = np.ma.masked_all((1,) + self.shape[1:])
            data_new[...] = eff_prod_capacity
            data_new = np.ma.concatenate([self.get_data(), data_new], axis=0)
            vars_new = np.concatenate([self.get_vars(), np.array(['effective_production_capacity'])])
            return AggrData(data_new, vars_new, self.data_capsule.regions,
                            self.data_capsule.sectors, self.data_capsule.lambda_axis,
                            self.data_capsule.duration_axis, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios,
                            _slope_meta=self.slope_meta)

    def calc_eff_forcing(self, _inplace=False):
        if 'effective_forcing' in self.get_vars():
            print('Variable \'effective_forcing\' already in data. Doing nothing.')
            return
        if 'forcing' not in self.get_vars() or 'production' not in self.get_vars():
            print('Variables \'forcing\' and \'production\' required to calculate effective forcing. Doing nothing.')
            return
        eff_forcing = np.ma.masked_all((1,) + self.shape[1:])
        for r_idx, sub_regions in enumerate(self.get_regions().values()):
            if len(sub_regions) > 1 and list(self.get_regions().keys())[r_idx] in sub_regions:
                sub_regions.remove(list(self.get_regions().keys())[r_idx])
            for s_idx, sub_sectors in enumerate(self.get_sectors().values()):
                if len(sub_sectors) > 1 and list(self.get_sectors().keys())[s_idx] in sub_sectors:
                    sub_sectors.remove(list(self.get_sectors().keys())[s_idx])
                original_forcings = self.get_regions(sub_regions).get_sectors(sub_sectors).get_vars(
                    'forcing').get_data()
                if len(sub_regions) > 1 or len(sub_sectors) > 1:
                    baseline_productions = self.get_regions(sub_regions).get_sectors(sub_sectors).get_vars(
                        'production').clip(1).get_data()
                    eff_forcing[0, r_idx, s_idx, ...] = np.sum(original_forcings * baseline_productions, axis=(1, 2),
                                                               keepdims=True) / np.sum(baseline_productions,
                                                                                       axis=(1, 2),
                                                                                       keepdims=True)
                else:
                    eff_forcing[0, r_idx, s_idx, ...] = original_forcings
        if _inplace:
            self.add_var(eff_forcing, 'effective_forcing', _inplace=True)
        else:
            return self.add_var(eff_forcing, 'effective_forcing', _inplace=False)

    def calc_demand_exceedence(self, _inplace=False):
        if 'demand_exceedence' in self.get_vars():
            print('Variable \'demand_exceedence\' already in data. Doing nothing.')
            return
        if 'effective_forcing' not in self.get_vars() or 'production' not in self.get_vars() or 'incoming_demand' not in self.get_vars():
            print(
                'Variables \'effective_forcing\', \'production\' and \'incoming_demand\' required to calculate demand_exceedence. Doing nothing.')
            return
        demand_exceedence = self.get_vars('incoming_demand').get_data() / (self.get_vars(
            'effective_forcing').get_data() * self.get_vars('production').clip(1).get_data()) - 1
        if _inplace:
            self.add_var(demand_exceedence, 'demand_exceedence', _inplace=True)
        else:
            data_new = np.ma.masked_all((1,) + self.shape[1:])
            data_new[...] = demand_exceedence
            data_new = np.ma.concatenate([self.get_data(), data_new], axis=0)
            vars_new = np.concatenate([self.get_vars(), np.array(['demand_exceedence'])])
            return AggrData(data_new, vars_new, self.data_capsule.regions,
                            self.data_capsule.sectors, self.data_capsule.lambda_axis,
                            self.data_capsule.duration_axis, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios,
                            _slope_meta=self.slope_meta)

    def calc_change_to_baseline(self, mode, _inplace=False, _aggregate=False):
        print('Calculating relative change to baseline values of all variables. Make sure that t=0 is the baseline'
              'state!')
        baseline_reference = self.get_data()[..., 0:1]
        if mode == 'relative':
            data_new = self.get_data() / self.get_data()[..., 0:1]
        elif mode == 'absolute':
            data_new = self.get_data() - self.get_data()[..., 0:1]
        if _aggregate:
            data_new = data_new.sum(axis=-1, keepdims=True)
            if mode == 'relative':
                aggregation_time = self.get_data().shape[-1]
                data_new /= aggregation_time
        vars_new = np.array([v + '_{}_change'.format(mode) for v in self.data_capsule.variables])
        if _inplace:
            self.data_capsule.data = data_new
            self.data_capsule.variables = vars_new
        else:
            return AggrData(data_new, vars_new, self.data_capsule.regions,
                            self.data_capsule.sectors, self.data_capsule.lambda_axis,
                            self.data_capsule.duration_axis, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios,
                            _slope_meta=self.slope_meta)

    def calc_demand_production_gap(self, _inplace=False):
        if 'demand_production_gap' in self.get_vars():
            print('Variable \'demand_production_gap\' already in data. Doing nothing.')
            return
        if 'effective_forcing' not in self.get_vars() or 'production' not in self.get_vars() or 'incoming_demand' not in self.get_vars():
            print(
                'Variables \'effective_forcing\', \'production\' and \'incoming_demand\' required to calculate demand_production_gap. Doing nothing.')
            return
        demand_prod_gap = self.get_vars('incoming_demand').get_data() - self.get_vars('production').get_data()
        if _inplace:
            self.add_var(demand_prod_gap, 'demand_production_gap', _inplace=True)
        else:
            data_new = np.ma.masked_all((1,) + self.shape[1:])
            data_new[...] = demand_prod_gap
            data_new = np.ma.concatenate([self.get_data(), data_new], axis=0)
            vars_new = np.concatenate([self.get_vars(), np.array(['demand_production_gap'])])
            return AggrData(data_new, vars_new, self.data_capsule.regions,
                            self.data_capsule.sectors, self.data_capsule.lambda_axis,
                            self.data_capsule.duration_axis, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios,
                            _slope_meta=self.slope_meta)

    def calc_desired_overproduction_capacity(self, _inplace=False):
        if 'desired_overproduction_capacity' in self.get_vars():
            print('Variable \'desired_overproduction_capacity\' already in data. Doing nothing.')
            return
        if 'desired_production_capacity' not in self.get_vars() or 'forcing' not in self.get_vars():
            print(
                'Variables \'desired_production_capacity\' and \'forcing\' required to calculate desired_overproduction_capacity. Doing nothing.')
            return
        des_overprod_capac = self.get_vars('desired_production_capacity').get_data() - self.get_vars(
            'forcing').get_data() + 1
        if _inplace:
            self.add_var(des_overprod_capac, 'desired_overproduction_capacity', _inplace=True)
        else:
            data_new = np.ma.masked_all((1,) + self.shape[1:])
            data_new[...] = des_overprod_capac
            data_new = np.ma.concatenate([self.get_data(), data_new], axis=0)
            vars_new = np.concatenate([self.get_vars(), np.array(['desired_overproduction_capacity'])])
            return AggrData(data_new, vars_new, self.data_capsule.regions,
                            self.data_capsule.sectors, self.data_capsule.lambda_axis,
                            self.data_capsule.duration_axis, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios,
                            _slope_meta=self.slope_meta)

    def aggregate(self, _method, _clip=None, _vars=None):
        if self.get_sim_duration() <= 1:
            raise ValueError('can only aggregate over more than one time step.')
        if _clip is None:
            _clip = self.get_sim_duration()
        if _vars is None:
            _vars = self.get_vars()
        data_sum = self.get_vars(_vars).clip(_clip).get_data().sum(axis=-1, keepdims=True)
        data_baseline = self.get_vars(_vars).clip(1).get_data() * _clip
        if _method == 'relative_difference':
            data_new = (data_sum / data_baseline - 1) * 100
        elif _method == 'absolute_difference':
            data_new = data_sum - data_baseline
        elif _method == 'sum':
            data_new = data_sum
        else:
            raise ValueError('{} is not a valid aggregation method.'.format(_method))
        vars_new = np.array([v + '_{}_{}'.format(_method, _clip) for v in self.data_capsule.variables])
        return AggrData(data_new, vars_new, self.data_capsule.regions,
                        self.data_capsule.sectors, self.data_capsule.lambda_axis,
                        self.data_capsule.duration_axis, _base_damage=self.base_damage,
                        _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios,
                        _slope_meta=self.slope_meta)

    def calc_sector_diff(self, sec1, sec2, _inplace=False):
        if sec1 not in self.get_sectors() or sec2 not in self.get_sectors():
            raise ValueError("Either {} or {} could not be found in sectors.".format(sec1, sec2))
        sec1_idx = np.where(np.array(list(self.get_sectors().keys())) == sec1)[0][0]
        sec2_idx = np.where(np.array(list(self.get_sectors().keys())) == sec2)[0][0]
        sec_data = self.data[:, :, sec1_idx:sec1_idx + 1, ...] - self.data[:, :, sec2_idx:sec2_idx + 1, ...]
        data_new = np.concatenate((self.data, sec_data), axis=2)
        new_sectors = copy.deepcopy(self.data_capsule.sectors)
        diff_sector_list = copy.deepcopy(new_sectors[sec1])
        diff_sector_list.remove(sec2)
        new_sectors["{}-{}".format(sec1, sec2)] = diff_sector_list
        if _inplace:
            self.data_capsule.data = data_new
            self.data_capsule.sectors = new_sectors
        else:
            return AggrData(data_new, self.data_capsule.variables, self.data_capsule.regions,
                            new_sectors, self.data_capsule.lambda_axis,
                            self.data_capsule.duration_axis, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios,
                            _slope_meta=self.slope_meta)

    def aggregate_regions(self, region_list, _name=None, _inplace=False):
        for r in region_list:
            if r not in self.get_regions():
                print("Warning. {} could not be found in regions.".format(r))
                region_list = list(set(region_list) - {r})
                if len(region_list) == 0:
                    raise ValueError("No regions left to aggregate.")
        r_data = np.nansum(self.get_regions(region_list).data, axis=1, keepdims=True)
        data_new = np.concatenate((self.data, r_data), axis=1)
        new_regions = copy.deepcopy(self.data_capsule.regions)
        if _name is None:
            _name = ''
            for r in region_list:
                _name += '{}+'.format(r)
            _name = _name[:-1]
        new_regions[_name] = region_list
        if _inplace:
            self.data_capsule.data = data_new
            self.data_capsule.regions = new_regions
        else:
            return AggrData(data_new, self.data_capsule.variables, new_regions,
                            self.data_capsule.sectors, self.data_capsule.lambda_axis,
                            self.data_capsule.duration_axis, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios,
                            _slope_meta=self.slope_meta)


def have_equal_shape(data1: AggrData, data2: AggrData):
    return (np.all(data1.get_vars() == data2.get_vars()) and
            np.all(data1.get_regions() == data1.get_regions()) and
            np.all(data1.get_sectors() == data2.get_sectors()) and
            np.all(data1.get_dt() == data2.get_dt()) and
            np.all(data1.get_re() == data2.get_re()) and
            np.all(data1.shape == data2.shape))


# v r s l d
def calc_dataset_stationarity_and_offset(_data: AggrData, **kwargs):
    df = pd.DataFrame(columns=['variable', 'region', 'sector', 'lambda', 'duration', 'num_seq', 'from', 'to', 'offset'])
    for v in _data.get_vars():
        for r in _data.get_regions():
            for s in _data.get_sectors():
                for l in _data.get_re_axis():
                    for d in _data.get_dt_axis():
                        ts = _data.get_vars(v).get_regions(r).get_sectors(s).get_re(l).get_dt(
                            d).get_data().flatten()
                        segment_and_offset, recursion_round = detect_stationarity_and_offset_in_series(ts, **kwargs)
                        if len(segment_and_offset) == 0:
                            print("Attention. No stationary segment found for var = {}, r = {}, s = {}, l = {}, "
                                  "d = {}".format(v, r, s, l, d))
                        elif recursion_round > 0:
                            print(
                                "Attention. Stationary segment for var = {}, r = {}, s = {}, l = {}, d = {} only found "
                                "in recurion round {} with _threshold = . Consider choosing a different _threshold "
                                "value next time".format(v, r, s, l, d, recursion_round,
                                                         kwargs.get('_threshold') * np.power(2, recursion_round)))
                        for seq_idx in range(len(segment_and_offset)):
                            df.loc[len(df)] = [v, r, s, l, d, seq_idx] + list(segment_and_offset[seq_idx])
    index = pd.MultiIndex.from_frame(df[['variable', 'region', 'sector', 'lambda', 'duration', 'num_seq']])
    columns = ['from', 'to', 'offset']
    df = pd.DataFrame(df[['from', 'to', 'offset']].to_numpy(), index=index, columns=columns).sort_index()
    return df


def clean_regions(_data: AggrData):
    for r in WORLD_REGIONS.keys():
        if r in _data.get_regions():
            _data.drop_region(r, _inplace=True)
        _data.aggregate_regions(list(set(WORLD_REGIONS[r]) - set(WORLD_REGIONS.keys())), _name=r, _inplace=True)


def load_from_xarray_data(data: xr.Dataset, baseline: xr.Dataset, sector_mapping: dict = None):
    acclimate_data = AcclimateOutput(data=data, baseline=baseline)


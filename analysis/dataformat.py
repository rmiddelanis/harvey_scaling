import numpy as np
from netCDF4 import Dataset

from utils import get_index_list


class DataCapsule:
    def __init__(self, _data: np.ndarray, _variables: np.ndarray, _regions: dict, _sectors: dict, ):
        if type(_data) is not np.ma.core.MaskedArray:
            raise ValueError("_data must be of type np.ndarray.")
        if type(_variables) is not np.ndarray:
            raise ValueError("_variables must be of type np.ndarray.")
        self.sectors = _sectors
        self.regions = _regions
        self.variables = _variables
        self.data = _data
        self.shape = _data.shape


# noinspection DuplicatedCode
class AggrData:
    def __init__(self, *args, _base_damage=None, _base_forcing=None, _scaled_scenarios=None):
        if len(args) == 1:
            if type(args[0]).__name__ == "DataCapsule":
                self.data_capsule = args[0]
            else:
                raise TypeError('Must pass argument of type DataCapsule or six arguments')
        elif len(args) == 4:
            self.data_capsule = DataCapsule(*args)
        else:
            raise TypeError('Must pass one argument of type DataCapsule or six arguments')
        self.base_damage = _base_damage
        self.base_forcing = _base_forcing
        self.scaled_scenarios = _scaled_scenarios

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
                            self.data_capsule.sectors, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios)

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
                            _base_damage=self.base_damage, _base_forcing=self.base_forcing,
                            _scaled_scenarios=self.scaled_scenarios)

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
                            _base_damage=self.base_damage, _base_forcing=self.base_forcing,
                            _scaled_scenarios=self.scaled_scenarios)

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
                        self.data_capsule.sectors, _base_damage=self.base_damage, _base_forcing=self.base_forcing,
                        _scaled_scenarios=self.scaled_scenarios)

    def get_data(self):
        return self.data_capsule.data

    def shape(self):
        return self.data_capsule.data.shape

    def get_sim_duration(self):
        return self.data_capsule.data.shape[-1]

    def add_var(self, _data, _name, _inplace=False):
        if _data.shape[1:-1] != self.get_data().shape[1:-1]:
            raise ValueError('Must pass array same dimensions in region, sector and duration!')
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
                            self.data_capsule.sectors, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios)

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
                            self.data_capsule.sectors, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios)

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
            data_new = np.ma.masked_all((len(add_vars),) + self.shape()[1:])
            vars_new = self.get_vars()
            for var_idx, var in enumerate(add_vars):
                data_new[var_idx, ...] = self.get_vars(var + "_value").get_data() / self.get_vars(var).get_data()
                vars_new = np.concatenate([vars_new, np.array([var + "_value"])])
            data_new = np.ma.concatenate([self.get_data(), data_new], axis=0)
            return AggrData(data_new, vars_new, self.data_capsule.regions,
                            self.data_capsule.sectors, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios)

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
            data_new = np.ma.masked_all((1,) + self.shape()[1:])
            data_new[...] = eff_prod_capacity
            data_new = np.ma.concatenate([self.get_data(), data_new], axis=0)
            vars_new = np.concatenate([self.get_vars(), np.array(['effective_production_capacity'])])
            return AggrData(data_new, vars_new, self.data_capsule.regions,
                            self.data_capsule.sectors, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios)

    def calc_eff_forcing(self, _inplace=False):
        if 'effective_forcing' in self.get_vars():
            print('Variable \'effective_forcing\' already in data. Doing nothing.')
            return
        if 'forcing' not in self.get_vars() or 'production' not in self.get_vars():
            print('Variables \'forcing\' and \'production\' required to calculate effective forcing. Doing nothing.')
            return
        eff_forcing = np.ma.masked_all((1,) + self.shape()[1:])
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
            data_new = np.ma.masked_all((1,) + self.shape()[1:])
            data_new[...] = demand_exceedence
            data_new = np.ma.concatenate([self.get_data(), data_new], axis=0)
            vars_new = np.concatenate([self.get_vars(), np.array(['demand_exceedence'])])
            return AggrData(data_new, vars_new, self.data_capsule.regions,
                            self.data_capsule.sectors, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios)

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
            data_new = np.ma.masked_all((1,) + self.shape()[1:])
            data_new[...] = demand_prod_gap
            data_new = np.ma.concatenate([self.get_data(), data_new], axis=0)
            vars_new = np.concatenate([self.get_vars(), np.array(['demand_production_gap'])])
            return AggrData(data_new, vars_new, self.data_capsule.regions,
                            self.data_capsule.sectors, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios)

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
            data_new = np.ma.masked_all((1,) + self.shape()[1:])
            data_new[...] = des_overprod_capac
            data_new = np.ma.concatenate([self.get_data(), data_new], axis=0)
            vars_new = np.concatenate([self.get_vars(), np.array(['desired_overproduction_capacity'])])
            return AggrData(data_new, vars_new, self.data_capsule.regions,
                            self.data_capsule.sectors, _base_damage=self.base_damage,
                            _base_forcing=self.base_forcing, _scaled_scenarios=self.scaled_scenarios)


def have_equal_shape(data1: AggrData, data2: AggrData):
    return (np.all(data1.get_vars() == data2.get_vars()) and
            np.all(data1.get_regions() == data1.get_regions()) and
            np.all(data1.get_sectors() == data2.get_sectors()) and
            np.all(data1.shape() == data2.shape()))


def read_file(_filename, _vars, _region_groups: dict, _sector_groups: dict, _time_frame):
    try:
        dataset = Dataset(_filename, "r", format="NETCDF4")
    except Exception as e:
        print("File {} could not be loaded.".format(_filename))
        print(e)
        return
    sim_duration = len(dataset.variables['time'])
    if _time_frame == -1:
        _time_frame = sim_duration
    elif sim_duration < _time_frame:
        print("Dataset {} contains less time steps ({}) than requested ({}). Continue with dataset length instead, "
              "mask remaining length.".format(_filename, sim_duration, _time_frame))
        _time_frame = sim_duration
    data_array = np.zeros((len(_vars), len(_region_groups), len(_sector_groups), _time_frame))
    mask_array = data_array != 0
    region_indices = {}
    sector_indices = {}
    for v, var in enumerate(_vars):
        _data = dataset["/agents/" + var.replace('/', '')][:].data
        _data = np.ma.array(_data, mask=np.isnan(_data), fill_value=0)
        for r, region_group_name in enumerate(_region_groups):
            if region_group_name in region_indices.keys():
                region_index = region_indices[region_group_name]
            else:
                region_index = get_index_list(_region_groups[region_group_name], dataset["/region"][:])
                region_indices[region_group_name] = region_index
            for s, sector_group_name in enumerate(_sector_groups):
                if sector_group_name in sector_indices.keys():
                    sector_index = sector_indices[sector_group_name]
                else:
                    sector_index = get_index_list(_sector_groups[sector_group_name], dataset["/sector"][:])
                    sector_indices[sector_group_name] = sector_index
                v_r_s_series = np.sum(np.sum(_data[:_time_frame, sector_index, :][:, :, region_index], axis=1), axis=1)
                data_array[v, r, s, :_time_frame] = v_r_s_series
                mask_array[v, r, s, _time_frame:] = data_array[v, r, s, _time_frame:] == 0
    dataset.close()
    return AggrData(np.ma.core.array(data_array, mask = mask_array), np.array(_vars), _region_groups, _sector_groups)
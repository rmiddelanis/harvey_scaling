import copy
import itertools
import os
import pickle
import numpy as np
import tqdm
import yaml

from scipy import signal
from netCDF4 import Dataset

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import pycountry as pc

WORLD_REGIONS = {
    'AFR': [
        'DZA', 'AGO', 'CPV', 'TZA', 'BWA', 'DJI', 'GIN', 'SYC', 'MAR', 'ZAF', 'BEN', 'CMR', 'LSO', 'TCD', 'MOZ', 'GNQ',
        'COG', 'ETH', 'MDG', 'RWA', 'ZMB', 'CAF', 'SOM', 'ERI', 'GAB', 'STP', 'EGY', 'NAM', 'GHA', 'LBR', 'LBY', 'BFA',
        'MRT', 'NGA', 'MWI', 'UGA', 'BDI', 'MUS', 'NER', 'SEN', 'GMB', 'ZWE', 'KEN', 'TUN', 'SWZ', 'CIV', 'TGO', 'MLI',
        'SLE', 'COD', 'SDN', 'SDS'
    ],
    'ASI': [
        'BGD', 'QAT', 'PAK', 'VNM', 'THA', 'NPL', 'YEM', 'PHL', 'SYR', 'MAC', 'GEO', 'TJK', 'PSE', 'IND', 'MDV', 'MMR',
        'RUS', 'KOR', 'IRQ', 'IRN', 'ARE', 'BHR', 'ARM', 'PNG', 'JOR', 'MYS', 'PRK', 'KHM', 'HKG', 'SAU', 'LBN', 'CHN',
        'KAZ', 'LKA', 'TKM', 'MNG', 'AFG', 'BTN', 'ISR', 'IDN', 'LAO', 'TUR', 'OMN', 'BRN', 'TWN', 'AZE', 'SGP', 'UZB',
        'KWT', 'JPN', 'KGZ'
    ],
    'EUR': [
        'BGR', 'FIN', 'ROU', 'BEL', 'GBR', 'HUN', 'BLR', 'GRC', 'AND', 'ANT', 'NOR', 'SMR', 'MDA', 'SRB', 'LTU', 'SWE',
        'AUT', 'ALB', 'MKD', 'UKR', 'CHE', 'LIE', 'PRT', 'SVN', 'SVK', 'HRV', 'DEU', 'NLD', 'MNE', 'LVA', 'IRL', 'CZE',
        'LUX', 'ISL', 'FRA', 'DNK', 'ITA', 'CYP', 'BIH', 'POL', 'EST', 'ESP', 'MLT', 'MCO'
    ],
    'LAM': [
        'NIC', 'GUY', 'CRI', 'TTO', 'PAN', 'BLZ', 'VGB', 'HND', 'DOM', 'PER', 'COL', 'VEN', 'MEX', 'ABW', 'ARG', 'BHS',
        'BOL', 'PRY', 'CHL', 'JAM', 'URY', 'HTI', 'ATG', 'SUR', 'ECU', 'GTM', 'CUB', 'BRB', 'BRA', 'CYM', 'SLV'
    ],
    'NMA': [
        'GRL', 'CAN', 'BMU', 'USA'
    ],
    'OCE': [
        'FJI', 'VUT', 'AUS', 'WSM', 'NZL', 'NCL', 'PYF'
    ],
    'ADB': [
        'BGD', 'PAK', 'VNM', 'THA', 'NPL', 'PHL', 'GEO', 'TJK', 'IND', 'MDV', 'MMR', 'KOR', 'ARE', 'ARM', 'PNG', 'MYS',
        'HKG', 'CHN', 'KAZ', 'TKM', 'MNG', 'AFG', 'BTN', 'IDN', 'LAO', 'TUR', 'TWN', 'AZE', 'UZB', 'JPN', 'KGZ'
    ],
    'EU28': [
        'BGR', 'FIN', 'ROU', 'BEL', 'GBR', 'HUN', 'GRC', 'LTU', 'SWE', 'AUT', 'PRT', 'SVN', 'SVK', 'HRV', 'DEU', 'NLD',
        'LVA', 'IRL', 'CZE', 'LUX', 'FRA', 'DNK', 'ITA', 'CYP', 'POL', 'EST', 'ESP', 'MLT'
    ],
    'OECD': [
        'AUS', 'AUT', 'BEL', 'CAN', 'CHE', 'CHL', 'CZE', 'DEU', 'DNK', 'ESP', 'EST', 'FIN', 'FRA', 'GBR', 'GRC', 'HUN',
        'IRL', 'ISL', 'ISR', 'ITA', 'JPN', 'KOR', 'LUX', 'LVA', 'MEX', 'NLD', 'NOR', 'NZL', 'POL', 'PRT', 'SVK', 'SVN',
        'SWE', 'TUR', 'USA'
    ],
    'BRICS': [
        'BRA', 'CHN', 'IND', 'RUS', 'ZAF'
    ],
    'CHN': [
        'CN.AH', 'CN.BJ', 'CN.CQ', 'CN.FJ', 'CN.GS', 'CN.GD', 'CN.GX', 'CN.GZ', 'CN.HA', 'CN.HB', 'CN.HL', 'CN.HE',
        'CN.HU', 'CN.HN', 'CN.JS', 'CN.JX', 'CN.JL', 'CN.LN', 'CN.NM', 'CN.NX', 'CN.QH', 'CN.SA', 'CN.SD', 'CN.SH',
        'CN.SX', 'CN.SC', 'CN.TJ', 'CN.XJ', 'CN.XZ', 'CN.YN', 'CN.ZJ'
    ],
    'USA': [
        'US.AL', 'US.AK', 'US.AZ', 'US.AR', 'US.CA', 'US.CO', 'US.CT', 'US.DE', 'US.DC', 'US.FL', 'US.GA', 'US.HI',
        'US.ID', 'US.IL', 'US.IN', 'US.IA', 'US.KS', 'US.KY', 'US.LA', 'US.ME', 'US.MD', 'US.MA', 'US.MI', 'US.MN',
        'US.MS', 'US.MO', 'US.MT', 'US.NE', 'US.NV', 'US.NH', 'US.NJ', 'US.NM', 'US.NY', 'US.NC', 'US.ND', 'US.OH',
        'US.OK', 'US.OR', 'US.PA', 'US.RI', 'US.SC', 'US.SD', 'US.TN', 'US.TX', 'US.UT', 'US.VT', 'US.VA', 'US.WA',
        'US.WV', 'US.WI', 'US.WY'
    ]
}

if 'WORLD' not in WORLD_REGIONS:
    WORLD_REGIONS['WORLD'] = list(set([ctry for wr in WORLD_REGIONS.values() for ctry in wr]) - set(WORLD_REGIONS.keys()))
if 'ROW' not in WORLD_REGIONS:
    WORLD_REGIONS['ROW'] = list(set(WORLD_REGIONS['WORLD']) - {'USA'} - set(WORLD_REGIONS['USA']))
if 'USA_REST_SANDY' not in WORLD_REGIONS:
    WORLD_REGIONS['USA_REST_SANDY'] = list(set(WORLD_REGIONS['USA']) - {'USA', 'US.NJ', 'US.NY'})
if 'USA_REST_HARVEY' not in WORLD_REGIONS:
    WORLD_REGIONS['USA_REST_HARVEY'] = list(set(WORLD_REGIONS['USA']) - {'USA', 'US.TX'})
WORLD_REGIONS['WORLD_wo_TX_LA'] = list(set(WORLD_REGIONS['WORLD']) - {'US.TX', 'US.LA'})
WORLD_REGIONS['MQ:25'] = ['RUS', 'CAN', 'VEN']
WORLD_REGIONS['MQ:50'] = ['RUS', 'CAN', 'VEN', 'AUS', 'NOR', 'IDN', 'DZA', 'SAU', 'IRN']
WORLD_REGIONS['MQ:60'] = ['RUS', 'CAN', 'VEN', 'AUS', 'NOR', 'IDN', 'DZA', 'SAU', 'IRN', 'CHN', 'USA', 'KWT']
WORLD_REGIONS['MQ:75'] = ['RUS', 'CAN', 'VEN', 'AUS', 'NOR', 'IDN', 'DZA', 'SAU', 'IRN', 'CHN', 'USA', 'KWT', 'GBR',
                          'BRA', 'NGA', 'AGO', 'ZAF', 'ARE', 'QAT', 'IND']
WORLD_REGIONS['MQ:95'] = ['RUS', 'CAN', 'VEN', 'AUS', 'NOR', 'IDN', 'DZA', 'SAU', 'IRN', 'CHN', 'USA', 'KWT', 'GBR',
                          'BRA', 'NGA', 'AGO', 'ZAF', 'ARE', 'QAT', 'IND', 'MEX', 'IRQ', 'NLD', 'OMN', 'MYS', 'TTO',
                          'DEU', 'BEL', 'LBY', 'KAZ', 'CHL', 'VNM', 'FRA', 'ARG', 'BOL', 'DNK', 'COL', 'ESP', 'ITA',
                          'PER', 'BRN', 'UKR', 'ECU', 'SYR', 'JPN', 'CHE', 'YEM']
WORLD_REGIONS['AI-MQ:50'] = ['DEU', 'CHN', 'USA', 'JPN', 'FRA', 'ITA', 'GBR']
WORLD_REGIONS['AI-MQ:75'] = ['DEU', 'CHN', 'USA', 'JPN', 'FRA', 'ITA', 'GBR', 'NLD', 'BEL', 'KOR', 'CAN', 'ESP', 'SGP',
                             'CHE', 'MEX', 'RUS', 'IND', 'MYS']
WORLD_REGIONS['AI-MQ:95'] = ['DEU', 'CHN', 'USA', 'JPN', 'FRA', 'ITA', 'GBR', 'NLD', 'BEL', 'KOR', 'CAN', 'ESP', 'SGP',
                             'CHE', 'MEX', 'RUS', 'IND', 'MYS', 'SWE', 'AUT', 'THA', 'AUS', 'BRA', 'HKG', 'IDN', 'IRL',
                             'CZE', 'TWN', 'DNK', 'POL', 'FIN', 'HUN', 'PHL', 'TUR', 'ZAF', 'NOR', 'ARG', 'ISR', 'ARE',
                             'PRT', 'SVK', 'NZL', 'SAU', 'CHL']
WORLD_REGIONS['AI:50'] = ['DEU', 'CHN', 'USA', 'JPN', 'FRA', 'ITA', 'GBR', 'NLD']
WORLD_REGIONS['AI:75'] = ['DEU', 'CHN', 'USA', 'JPN', 'FRA', 'ITA', 'GBR', 'NLD', 'CAN', 'BEL', 'KOR', 'ESP', 'RUS',
                              'SGP', 'CHE', 'MEX', 'IND', 'MYS', 'AUS', 'SWE']
WORLD_REGIONS['AI:95'] = ['DEU', 'CHN', 'USA', 'JPN', 'FRA', 'ITA', 'GBR', 'NLD', 'CAN', 'BEL', 'KOR', 'ESP', 'RUS',
                              'SGP', 'CHE', 'MEX', 'IND', 'MYS', 'AUS', 'SWE', 'AUT', 'IDN', 'THA', 'BRA', 'HKG', 'IRL',
                              'CZE', 'DNK', 'TWN', 'POL', 'NOR', 'FIN', 'PHL', 'HUN', 'ZAF', 'VEN', 'TUR', 'SAU', 'ARE',
                              'ARG', 'ISR', 'IRN', 'PRT', 'CHL', 'SVK', 'NZL', 'UKR', 'DZA', 'KWT']
WORLD_REGIONS['TX_trade:50'] = ['CAN', 'MEX', 'JPN', 'DEU', 'GBR', 'VEN', 'KOR']
WORLD_REGIONS['TX_trade:75'] = ['CAN', 'MEX', 'JPN', 'DEU', 'GBR', 'VEN', 'KOR', 'FRA', 'ITA', 'BRA', 'SGP', 'NLD',
                                'AUS', 'IND', 'IRL', 'CHE', 'MYS', 'CN.GD', 'BEL', 'HKG', 'CN.JS', 'TWN', 'THA', 'CN.SD']
WORLD_REGIONS['TX_trade:95'] = ['CAN', 'MEX', 'JPN', 'DEU', 'GBR', 'VEN', 'KOR', 'FRA', 'ITA', 'BRA', 'SGP', 'NLD',
                                'AUS', 'IND', 'IRL', 'CHE', 'MYS', 'CN.GD', 'BEL', 'HKG', 'CN.JS', 'TWN', 'THA',
                                'CN.SD', 'ESP', 'SAU', 'GUY', 'IDN', 'ISR', 'PHL', 'SWE', 'CN.ZJ', 'RUS', 'COL',
                                'CN.HE', 'TTO', 'NGA', 'CN.SC', 'CN.HB', 'CN.HU', 'CN.HN', 'CN.LN', 'CHL', 'AUT',
                                'DZA', 'ARG', 'CN.FJ', 'ZAF', 'DNK', 'CN.SH', 'TUR', 'NOR', 'CN.BJ', 'CN.AH', 'KWT',
                                'FIN', 'CN.SA', 'CN.NM', 'DOM', 'CN.GX', 'CN.JX', 'CN.TJ', 'NZL', 'AGO', 'CN.CQ', 'ARE',
                                'HUN', 'CN.HL', 'CRI']
WORLD_REGIONS['EXPORT:50'] = ['DEU', 'CHN', 'USA', 'JPN', 'FRA', 'ITA', 'GBR', 'NLD']
WORLD_REGIONS['EXPORT:75'] = ['DEU', 'CHN', 'USA', 'JPN', 'FRA', 'ITA', 'GBR', 'NLD', 'CAN', 'BEL', 'KOR', 'ESP', 'RUS',
                              'SGP', 'CHE', 'MEX', 'IND', 'MYS', 'AUS', 'SWE']
WORLD_REGIONS['EXPORT:95'] = ['DEU', 'CHN', 'USA', 'JPN', 'FRA', 'ITA', 'GBR', 'NLD', 'CAN', 'BEL', 'KOR', 'ESP', 'RUS',
                              'SGP', 'CHE', 'MEX', 'IND', 'MYS', 'AUS', 'SWE', 'AUT', 'IDN', 'THA', 'BRA', 'HKG', 'IRL',
                              'CZE', 'DNK', 'TWN', 'POL', 'NOR', 'FIN', 'PHL', 'HUN', 'ZAF', 'VEN', 'TUR', 'SAU', 'ARE',
                              'ARG', 'ISR', 'IRN', 'PRT', 'CHL', 'SVK', 'NZL', 'UKR', 'DZA']
for wr in ['MQ:25', 'MQ:50', 'MQ:60', 'MQ:75', 'MQ:95', 'TX_trade:50', 'TX_trade:75', 'TX_trade:95', 'AI-MQ:50',
           'AI-MQ:75', 'AI-MQ:95', 'AI:50', 'AI:75', 'AI:95', 'EXPORT:50', 'EXPORT:75', 'EXPORT:95']:
    for sr in list(WORLD_REGIONS[wr]):
        if sr in WORLD_REGIONS.keys():
            WORLD_REGIONS[wr].remove(sr)
            WORLD_REGIONS[wr] = list(set(WORLD_REGIONS[wr] + WORLD_REGIONS[sr]) - {sr})
WORLD_REGIONS['MQ:50+USA'] = WORLD_REGIONS['MQ:50'] + list(set(WORLD_REGIONS['USA']) - {'USA'})

SECTOR_GROUPS = {
    'ALLSECTORS': [
        'FCON', 'AGRI', 'FISH', 'MINQ', 'FOOD', 'TEXL', 'WOOD', 'OILC', 'METL', 'MACH', 'TREQ', 'MANU', 'RECY', 'ELWA',
        'CONS', 'REPA', 'WHOT', 'RETT', 'GAST', 'TRAN', 'COMM', 'FINC', 'ADMI', 'EDHE', 'HOUS', 'OTHE', 'REXI'
    ],
    'PRIVSECTORS': [
        'AGRI', 'FISH', 'MINQ', 'FOOD', 'TEXL', 'WOOD', 'OILC', 'METL', 'MACH', 'TREQ', 'MANU', 'RECY', 'ELWA', 'CONS',
        'REPA', 'WHOT', 'RETT', 'GAST', 'TRAN', 'COMM', 'FINC', 'ADMI', 'EDHE', 'HOUS', 'OTHE', 'REXI'
    ],
    'ALL_INDUSTRY': [
        'AGRI', 'FISH', 'MINQ', 'FOOD', 'TEXL', 'WOOD', 'OILC', 'METL', 'MACH', 'TREQ', 'MANU', 'RECY', 'ELWA', 'CONS',
        'REPA', 'WHOT', 'RETT', 'GAST', 'TRAN', 'COMM', 'FINC', 'ADMI', 'EDHE', 'HOUS', 'OTHE', 'REXI'
    ]
}

state_gdp_path = "~/repos/harvey_scaling/data/external/CAGDP2__ALL_AREAS_2001_2019.csv"

cn_gadm_to_iso_code = {
    'CN.BJ': 'CN.BJ',
    'CN.TJ': 'CN.TJ',
    'CN.HB': 'CN.HE',
    'CN.SX': 'CN.SX',
    'CN.NM': 'CN.NM',
    'CN.LN': 'CN.LN',
    'CN.JL': 'CN.JL',
    'CN.HL': 'CN.HL',
    'CN.SH': 'CN.SH',
    'CN.JS': 'CN.JS',
    'CN.ZJ': 'CN.ZJ',
    'CN.AH': 'CN.AH',
    'CN.FJ': 'CN.FJ',
    'CN.JX': 'CN.JX',
    'CN.SD': 'CN.SD',
    'CN.HE': 'CN.HA',
    'CN.HU': 'CN.HB',
    'CN.HN': 'CN.HN',
    'CN.GD': 'CN.GD',
    'CN.GX': 'CN.GX',
    'CN.HA': 'CN.HI',
    'CN.CQ': 'CN.CQ',
    'CN.SC': 'CN.SC',
    'CN.GZ': 'CN.GZ',
    'CN.YN': 'CN.YN',
    'CN.XZ': 'CN.XZ',
    'CN.SA': 'CN.SN',
    'CN.GS': 'CN.GS',
    'CN.QH': 'CN.QH',
    'CN.NX': 'CN.NX',
    'CN.XJ': 'CN.XJ'
}


def find_attribute_in_yaml(_yaml_root, _attribute_name):
    result = []
    if type(_yaml_root) is dict:
        for key in _yaml_root.keys():
            if key == _attribute_name:
                result.append(_yaml_root.get(key))
            elif type(_yaml_root.get(key)) is list or dict:
                recursion_results = find_attribute_in_yaml(_yaml_root.get(key), _attribute_name)
                for recursion_result in recursion_results:
                    if recursion_result not in result:
                        result.append(recursion_result)
    elif type(_yaml_root) is list:
        for child in _yaml_root:
            if type(child) is not str:
                recursion_results = find_attribute_in_yaml(child, _attribute_name)
                for recursion_result in recursion_results:
                    if recursion_result not in result:
                        result.append(recursion_result)
    return result


def get_axes_and_dirs(_root_dir):
    if os.path.exists(os.path.join(_root_dir, "ensemble_meta.pk")):
        _ensemble_meta = pickle.load(open(os.path.join(_root_dir, "ensemble_meta.pk"), 'rb'))
        # ensemble_meta_temp = {}
        # for key, val in _ensemble_meta.items():
        #     ensemble_meta_temp[(np.round(key[0], 3), np.round(key[1], 3))] = val
        # _ensemble_meta = ensemble_meta_temp
        _lambda_axis = set([key[0] for key in _ensemble_meta['scaled_scenarios'].keys()])
        _duration_axis = set([key[1] for key in _ensemble_meta['scaled_scenarios'].keys()])
    else:
        _lambda_axis = []
        _duration_axis = []
        _ensemble_meta = {}
        for name in os.listdir(_root_dir)[:]:
            iter_dir = os.path.join(_root_dir, name)
            if os.path.isdir(iter_dir):
                try:
                    settings_file = open(r'' + iter_dir + '/settings.yml', 'r')
                except Exception as e:
                    print("{} does not contain a settings.yml file".format(iter_dir))
                    print(e)
                    continue
                iter_settings = yaml.load(settings_file, Loader=yaml.FullLoader)
                iter_lambda = find_attribute_in_yaml(iter_settings, 'remaining_capacity')
                iter_lambda = iter_lambda[0] if len(iter_lambda) == 1 else print("len(iter_lambda)>1")
                iter_duration_from = find_attribute_in_yaml(iter_settings, 'from')
                if len(iter_duration_from) >= 1:
                    if len(iter_duration_from) > 1:
                        print(
                            "Attention. More than one event found: iter_duration_from={} Considering only the first one.".format(
                                iter_duration_from))
                    iter_duration_from = iter_duration_from[0]
                else:
                    print("Attention. No event found. iter_duration_from: {}".format(iter_duration_from))
                iter_duration_to = find_attribute_in_yaml(iter_settings, 'to')
                if len(iter_duration_to) >= 1:
                    if len(iter_duration_to) > 1:
                        print(
                            "Attention. More than one event found: iter_duration_to={} Considering only the first one.".format(
                                iter_duration_to))
                    iter_duration_to = iter_duration_to[0]
                else:
                    print("Attention. No event found. iter_duration_to: {}".format(iter_duration_to))
                iter_duration = iter_duration_to - iter_duration_from + 1
                if iter_duration not in _duration_axis:
                    _duration_axis.append(iter_duration)
                if iter_lambda not in _lambda_axis:
                    _lambda_axis.append(iter_lambda)
                _ensemble_meta[(iter_lambda, iter_duration)] = {'iter_name': name}
    return np.array(sorted(_lambda_axis)), np.array(sorted(_duration_axis)), _ensemble_meta


def get_index(_item, _array):
    if type(_array) is list:
        try:
            return _array.index(_item)
        except ValueError:
            return None
    elif type(_array) is np.ndarray:
        _index, = np.where(_array == _item)
        if len(_index) > 0:
            return _index[0]
        else:
            return None


def get_index_list(_array_origin, _array_target):
    if type(_array_origin) is not list and type(_array_origin) is not np.ndarray:
        _array_origin = [_array_origin]
    if not (type(_array_target) is list or type(_array_target) is np.ndarray):
        raise TypeError("_array_target should be of type list or numpy.ndarray")
    _result = []
    for origin_item in _array_origin:
        index = get_index(origin_item, _array_target)
        if index is not None:
            _result.append(index)
        else:
            print("Attention. Item \"{}\" could not be found in target array.".format(origin_item))
    return _result


def pickle_data(data, datatype, _experiment_series_dir, filename=''):
    pickle_dir = os.path.join(_experiment_series_dir, "pickles/")
    if not os.path.exists(pickle_dir):
        os.mkdir(pickle_dir)
        print("Directory ", pickle_dir, " Created ")
    else:
        print("Directory ", pickle_dir, " already exists")
    if not (type(filename) is str and len(filename) > 0):
        filename = dt.datetime.now()
    file = os.path.join(pickle_dir, "{}__{}.pk".format(filename, datatype))
    try:
        pickle.dump(data, open(file, 'wb'))
        print("Saved as {}".format(file))
    except Exception as e:
        print(e)


def load_pickle_data(filename: str, _experiment_series_dir, pickle_folder_name=''):
    if filename[-3:] != '.pk':
        filename = filename + '.pk'
    if len(pickle_folder_name) == 0:
        pickle_folder_name = os.path.join(_experiment_series_dir, "pickles/")
    loaded_data = pickle.load(open(os.path.join(pickle_folder_name, filename), "rb"))
    return loaded_data


def get_regions_dict(_regions):
    result = {}
    for region in _regions:
        try:
            result[region] = WORLD_REGIONS[region]
        except KeyError:
            if region in WORLD_REGIONS['WORLD']:
                result[region] = [region]
                print("Key {} was not found in world regions. Adding as a single country instead".format(region))
            else:
                print("Key {} is neither a world region nor a country. Ignoring key".format(region))
    return result


def get_sectors_dict(_sectors):
    result = {}
    for sector in _sectors:
        try:
            result[sector] = SECTOR_GROUPS[sector]
        except KeyError:
            if sector in SECTOR_GROUPS['ALLSECTORS']:
                result[sector] = [sector]
                print("Key {} was not found in sector groups. Adding as a single sector instead".format(sector))
            else:
                print("Key {} is neither a sector group nor a single sector. Ignoring key".format(sector))
    return result


def make_figure_dir(_experiment_series_dir):
    figure_dir = os.path.join(_experiment_series_dir, "figures")
    subdirs = ['heatmaps', 'choropleth', 'time_series']
    if not os.path.exists(figure_dir):
        for subdir in subdirs:
            os.makedirs(os.path.join(figure_dir, subdir))
    else:
        for subdir in subdirs:
            if not os.path.exists(os.path.join(figure_dir, subdir)):
                os.mkdir(os.path.join(figure_dir, subdir))
    print("Figure directories set up for {}.".format(_experiment_series_dir))


def detect_stationarity_and_offset_in_series(time_series_masked, _threshold=0.001, window_size=50, max_segments=None,
                                             sortby='longest', allow_empty_result=True, make_plot=False, recursion=0):
    time_series = time_series_masked[~time_series_masked.mask]
    if len(time_series) == 0:
        print("Entire series is masked. Stationarity calculatino aborted.")
        return [], recursion
    if np.var(time_series) == 0:
        return [(0, len(time_series) - 1, 0)], recursion
    baseline_value = time_series[0]
    variances = []
    for step in range(len(time_series) - window_size):
        chunk = time_series[step:step + window_size]
        variances.append(np.sum(np.square(chunk - np.mean(chunk))))
    threshold_curve = np.where(variances < _threshold * np.mean(variances), 0, 1)
    stationary_segments = []
    segment_start = None
    if threshold_curve[0] == 0:
        segment_start = 0
    for idx, val in enumerate(threshold_curve[:-1]):
        if val > threshold_curve[idx + 1]:
            segment_start = idx + 1
        if segment_start is not None:
            if val < threshold_curve[idx + 1] or idx == len(threshold_curve) - 2:
                stationary_segment_mean = np.mean(time_series[segment_start:idx + 1])
                stationary_segment_offset = stationary_segment_mean - baseline_value
                stationary_segments.append((segment_start, idx + 1 + window_size, stationary_segment_offset))
                segment_start = None
    if not allow_empty_result and len(stationary_segments) == 0:
        print('Attention. No stationary segment found. Retry with double threshold. Recursion round '.format(
            recursion + 1))
        return detect_stationarity_and_offset_in_series(time_series, _threshold=2 * _threshold, window_size=window_size,
                                                        max_segments=max_segments,
                                                        allow_empty_result=allow_empty_result,
                                                        make_plot=make_plot, recursion=recursion + 1)
    if max_segments is not None:
        if sortby == 'longest':
            def sortkey(x):
                return x[1] - x[0]

            reverse = True
        if sortby == 'last':
            def sortkey(x):
                return x[1]

            reverse = True
        elif sortby == 'first':
            def sortkey(x):
                return x[0]

            reverse = True
        stationary_segments = sorted(stationary_segments, key=sortkey, reverse=reverse)
        stationary_segments = stationary_segments[:min(max_segments, len(stationary_segments))]
    stationary_segments = sorted(stationary_segments, key=lambda x: x[0])
    if make_plot:
        if len(stationary_segments) > 0:
            cp_indices = sorted(np.array(stationary_segments)[:, :1].flatten().astype(int))
            # noinspection PyTypeChecker
            fig, ax = plt.subplots(2, 1, sharex=True)
            ax[0].plot(time_series)
            ax[0].plot(cp_indices, time_series[cp_indices], marker='o', color='r', linestyle='')
            ax[1].plot(threshold_curve)
            plt.show()
        else:
            print("No stationary segments found to print for this time series.")
    return stationary_segments, recursion


def generate_dataframe(_data):
    if _data.get_sim_duration() > 1:
        print("Can only generate dataframe for aggregated datasets. Please make sure that the time dimension of the "
              "dataset has length 1.")
        return
    index = pd.MultiIndex.from_product([_data.get_regions(), _data.get_sectors(), _data.get_re_axis(),
                                        _data.get_dt_axis()])
    _df = pd.DataFrame(data=_data.get_data().reshape(len(_data.get_vars()), -1).transpose(), columns=_data.get_vars(),
                       index=index)
    return _df


def calc_ts_characteristic(_ts: np.ndarray, _method, _choose_max=False, _choose_min=False, _sortby='index',
                           _selection_index=0, _choose_index=False, _choose_value=False, _diff_to_baseline=False,
                           _window_from=None, _window_to=None, **peaks_kwargs):
    if _window_to is None:
        _window_to = len(_ts)
    if _window_from is None:
        _window_from = 0
    _ts_frame = _ts[_window_from:_window_to]
    if _method == 'min_value':
        characteristic = _ts_frame.min()
    if _method == 'min_timestep':
        characteristic = np.where[_ts_frame == _ts_frame.min()][0] + _window_from
    elif _method == 'max_value':
        characteristic = _ts_frame.max()
    elif _method == 'max_timestep':
        characteristic = np.where[_ts_frame == _ts_frame.max()][0] + _window_from
    elif _method == 'variability':
        characteristic = _ts_frame.max() - _ts_frame.min()
    elif _method == 'local_min_timestep':
        max_peaks, max_peak_props, min_peaks, min_peak_props = find_peaks_in_ts(_ts_frame, _order='height',
                                                                                **peaks_kwargs)
        characteristic = min_peaks[0] + _window_from
    elif _method == 'local_min_value':
        max_peaks, max_peak_props, min_peaks, min_peak_props = find_peaks_in_ts(_ts_frame, _order='height',
                                                                                **peaks_kwargs)
        characteristic = _ts_frame[min_peaks[0]]
    elif _method == 'local_max_timestep':
        max_peaks, max_peak_props, min_peaks, min_peak_props = find_peaks_in_ts(_ts_frame, _order='height',
                                                                                **peaks_kwargs)
        characteristic = max_peaks[0] + _window_from
    elif _method == 'local_max_value':
        max_peaks, max_peak_props, min_peaks, min_peak_props = find_peaks_in_ts(_ts_frame, _order='height',
                                                                                **peaks_kwargs)
        characteristic = _ts_frame[max_peaks[0]]
    elif _method == 'inter_peak_drop_intensity_rel':
        max_peaks, max_peak_props, min_peaks, min_peak_props = find_peaks_in_ts(_ts_frame, _order='height',
                                                                                **peaks_kwargs)
        characteristic = (_ts_frame[max_peaks[0]] - _ts_frame[min_peaks[0]]) / _ts_frame[max_peaks[0]]
    elif _method == 'inter_peak_drop_intensity_abs':
        max_peaks, max_peak_props, min_peaks, min_peak_props = find_peaks_in_ts(_ts_frame, _order='height',
                                                                                **peaks_kwargs)
        characteristic = (_ts_frame[max_peaks[0]] - _ts_frame[min_peaks[0]])
    elif _method == 'peaks':
        if _choose_max == _choose_min:
            raise ValueError("_choose_min and _choose_max must be different boolean values")
        if _choose_index == _choose_value:
            raise ValueError("_choose_index and _choose_value must be different boolean values")
        max_peaks, max_peak_props, min_peaks, min_peak_props = find_peaks_in_ts(_ts_frame, **peaks_kwargs)
        if len(max_peaks) == 0 and _choose_max:
            raise Warning("Attention. No Maxima found.")
        if len(min_peaks) == 0 and _choose_min:
            raise Warning("Attention. No Minima found.")
        if _sortby == 'prominence':
            max_peaks = max_peaks[max_peak_props['prominences'].argsort()]
            max_peak_props['prominences'] = max_peak_props['prominences'].argsort()
            min_peaks = min_peaks[min_peak_props['prominences'].argsort()]
            min_peak_props['prominences'] = min_peak_props['prominences'].argsort()
        elif _sortby == 'value':
            max_peaks = max_peaks[_ts_frame[max_peaks].argsort()]
            max_peak_props['prominences'] = max_peak_props['prominences'][_ts_frame[max_peaks].argsort()]
            min_peaks = min_peaks[_ts_frame[min_peaks].argsort()]
            min_peak_props['prominences'] = min_peak_props['prominences'][_ts_frame[min_peaks].argsort()]
        if _choose_max:
            if _choose_index:
                characteristic = max_peaks[_selection_index]
            elif _choose_value:
                if _diff_to_baseline:
                    characteristic = _ts_frame[0] - _ts_frame[max_peaks[_selection_index]]
                else:
                    characteristic = _ts_frame[max_peaks[_selection_index]]
        elif _choose_min:
            if _choose_index:
                characteristic = min_peaks[_selection_index]
            elif _choose_value:
                if _diff_to_baseline:
                    characteristic = _ts_frame[0] - _ts_frame[min_peaks[_selection_index]]
                else:
                    characteristic = _ts_frame[min_peaks[_selection_index]]
    elif _method == 'most_prominent_max_peak_value':
        max_peaks, max_peak_props, min_peaks, min_peak_props = find_peaks_in_ts(_ts_frame, _order='prominence',
                                                                                **peaks_kwargs)
        if len(max_peaks) >= 1:
            characteristic = _ts_frame[max_peaks[0]]
    elif _method == 'most_prominent_max_peak_timestep':
        max_peaks, max_peak_props, min_peaks, min_peak_props = find_peaks_in_ts(_ts_frame, _order='prominence',
                                                                                **peaks_kwargs)
        if len(max_peaks) >= 1:
            characteristic = max_peaks[0] + _window_from
    elif _method == 'first_maximum_value':
        max_peaks, max_peak_props, min_peaks, min_peak_props = find_peaks_in_ts(_ts_frame, **peaks_kwargs)
        if len(max_peaks) >= 1:
            characteristic = _ts_frame[max_peaks[0]]
    elif _method == 'first_maximum_timestep':
        max_peaks, max_peak_props, min_peaks, min_peak_props = find_peaks_in_ts(_ts_frame, **peaks_kwargs)
        if len(max_peaks) >= 1:
            characteristic = max_peaks[0] + _window_from
    elif _method == 'second_maximum_value':
        max_peaks, max_peak_props, min_peaks, min_peak_props = find_peaks_in_ts(_ts_frame, **peaks_kwargs)
        if len(max_peaks) >= 2:
            characteristic = _ts_frame[max_peaks[1]]
    elif _method == 'second_minimum_value':
        max_peaks, max_peak_props, min_peaks, min_peak_props = find_peaks_in_ts(_ts_frame, **peaks_kwargs)
        if len(min_peaks) >= 2:
            characteristic = _ts_frame[min_peaks[1]]
    elif _method == 'first_minimum_value':
        max_peaks, max_peak_props, min_peaks, min_peak_props = find_peaks_in_ts(_ts_frame, **peaks_kwargs)
        if len(min_peaks) >= 2:
            characteristic = _ts_frame[min_peaks[0]]
    elif _method[:7] == 'value_t':
        _t = int(_method[7:])
        characteristic = _ts[_t]
    elif _method[:10] == 'drop_ratio':
        _t_0 = int(_method.split('_')[2])
        _t_1 = int(_method.split('_')[3])
        _t_2 = int(_method.split('_')[4])
        max_peaks, max_peak_props, min_peaks, min_peak_props = find_peaks_in_ts(_ts_frame, _order='height', _from=_t_0,
                                                                                _to=_t_1)
        first_peak = max(_ts_frame[0], _ts_frame[max_peaks[0]]) if len(max_peaks) > 0 else _ts_frame[0]
        first_peak_drop = first_peak - _ts_frame[min_peaks[0]]
        max_peaks, max_peak_props, min_peaks, min_peak_props = find_peaks_in_ts(_ts_frame, _order='height', _from=_t_1,
                                                                                _to=_t_2)
        second_peak_drop = _ts_frame[max_peaks[0]] - _ts_frame[min_peaks[0]]
        characteristic = second_peak_drop / first_peak_drop
    elif _method == 'first_drop_intensity':
        max_peaks, max_peak_props, min_peaks, min_peak_props = find_peaks_in_ts(_ts_frame, _order='height')
        first_peak = max(_ts_frame[0], _ts_frame[max_peaks[0]]) if len(max_peaks) > 0 else _ts_frame[0]
        first_peak_drop = first_peak - _ts_frame[min_peaks[0]]
        characteristic = first_peak_drop
    return characteristic


def find_peaks_in_ts(_ts: np.ndarray, _from=None, _to=None, _order='index', **peak_kwargs):
    if _to is None:
        _to = len(_ts)
    if _from is None:
        _from = 0
    max_peaks, max_peak_props = signal.find_peaks(_ts[_from:_to], prominence=0.0001, **peak_kwargs)
    min_peaks, min_peak_props = signal.find_peaks(_ts[_from:_to] * -1, prominence=0.0001, **peak_kwargs)
    max_peaks = max_peaks + _from
    min_peaks = min_peaks + _from
    if _order == 'index':
        max_peaks_order = np.arange(len(max_peaks))
        min_peaks_order = np.arange(len(min_peaks))
    if _order == 'prominence':
        max_peaks_order = max_peak_props['prominences'].argsort()[::-1]
        min_peaks_order = min_peak_props['prominences'].argsort()[::-1]
    if _order == 'height':
        max_peaks_order = _ts[max_peaks].argsort()[::-1]
        min_peaks_order = _ts[min_peaks].argsort()
    max_peaks = max_peaks[max_peaks_order]
    max_peak_props['prominences'] = max_peak_props['prominences'][max_peaks_order]
    max_peak_props['left_bases'] = max_peak_props['left_bases'][max_peaks_order]
    max_peak_props['right_bases'] = max_peak_props['right_bases'][max_peaks_order]
    min_peaks = min_peaks[min_peaks_order]
    min_peak_props['prominences'] = min_peak_props['prominences'][min_peaks_order]
    min_peak_props['left_bases'] = min_peak_props['left_bases'][min_peaks_order]
    min_peak_props['right_bases'] = min_peak_props['right_bases'][min_peaks_order]
    return max_peaks, max_peak_props, min_peaks, min_peak_props


def calc_mean_f0(_data, _year, _states, _inplace=False):
    if _data.scaled_scenarios is None:
        raise ValueError('data must contain scaled scenarios meta information.')
    if _inplace:
        res = _data
    else:
        res = copy.deepcopy(_data)
    us_states_mapping = {sd.name: sd.code.replace('US-', '') for sd in pc.subdivisions if sd.country_code=='US'}
    gdp_by_state = pd.read_csv(state_gdp_path)
    gdp_by_state.drop(gdp_by_state.index[-4:], inplace=True)
    gdp_by_state = gdp_by_state[gdp_by_state['Description'] == 'All industry total']
    gdp_by_state = gdp_by_state[gdp_by_state['GeoName'].isin(us_states_mapping.keys())]
    gdp_by_state['GeoName'] = gdp_by_state['GeoName'].apply(lambda x: 'US.' + us_states_mapping[x])
    gdp_by_state = gdp_by_state[gdp_by_state['GeoName'].isin(_states)]
    gdp_by_state = gdp_by_state[['GeoName', str(_year)]].astype({'GeoName': str, str(_year): int})
    gdp_by_state.set_index('GeoName', inplace=True)
    total_gdp = gdp_by_state[str(_year)].sum()
    for (scaled_l, scaled_d) in list(itertools.product(_data.get_re_axis(), _data.get_dt_axis())):
        exposed_gdp_scaled = 0
        for state in _states:
            state_f0 = _data.scaled_scenarios[(scaled_l, scaled_d)]['params'][state]['f_0']
            exposed_gdp_scaled += gdp_by_state.loc[state, str(_year)] * state_f0
        mean_f0 = exposed_gdp_scaled / total_gdp
        res.scaled_scenarios[(scaled_l, scaled_d)]['params']['all']['f_0'] = mean_f0
    if not _inplace:
        return res

def set_direct_loss_meta(_data, _inplace=False):
    if _data.scaled_scenarios is None:
        raise ValueError('data must contain scaled scenarios meta information.')
    if 'direct_loss' not in _data.get_vars():
        raise ValueError('data must contain variable direct_loss')
    if _inplace:
        res = _data
    else:
        res = copy.deepcopy(_data)
    print("Direct loss will be calculated for {} sectors and {} regions".format(len(_data.get_vars()),
                                                                                len(_data.get_regions())))
    direct_loss = _data.get_vars('direct_loss').aggregate('sum')
    for (scaled_l, scaled_d) in list(itertools.product(_data.get_re_axis(), _data.get_dt_axis())):
        direct_loss_iter = direct_loss.get_re(scaled_l).get_dt(scaled_d).get_data().sum()
        res.scaled_scenarios[(scaled_l, scaled_d)]['params']['all']['direct_loss'] = direct_loss_iter
    if not _inplace:
        return res


def calc_direct_production_loss(_data, _inplace=False):
    if _inplace:
        res = _data
    else:
        res = copy.deepcopy(_data)
    days = np.arange(_data.get_sim_duration())
    for scenario_key, scenario in _data.scaled_scenarios.items():
        t_r = scenario['params']['all']['t_r']
        f_r = scenario['params']['all']['f_r']
        total_direct_loss = 0
        affected_states = list(scenario['params'].keys())[:-1]
        for state in affected_states:
            f0 = scenario['params'][state]['f_0']
            tau = scenario['params'][state]['tau']
            f = f0 * np.exp(-days / tau)
            f[t_r + 1:] = 0
            direct_loss = f * _data.get_vars('production').get_regions(affected_states)
    if not _inplace:
        return res



def calc_sector_export(_sectors='MINQ', _aggregate_chn_usa=True, _exclude_domestic_trade=True,
                       _baseline_path="/mnt/cluster_p/projects/acclimate/data/eora/Eora26-v199.82-2015-CHN-USA_naics_disagg.nc"):
    baseline_data = Dataset(_baseline_path)
    if type(_sectors) == str:
        _sectors = [_sectors]
    elif _sectors is None:
        _sectors = baseline_data['sector'][:]
    flows = baseline_data['flows'][:]
    exports = pd.DataFrame(columns=['export'])
    sec_indices = np.where(np.isin(baseline_data['sector'][:], _sectors))[0]
    for r in tqdm.tqdm(baseline_data['region'][:]):
        region_idx = np.where(baseline_data['region'][:] == r)[0][0]
        rs_from = np.where((baseline_data['index_region'][:] == region_idx) & (np.isin(baseline_data['index_sector'], sec_indices)))[0]
        excluded_region_indices = [region_idx]
        if _exclude_domestic_trade:
            if r[:3] == 'US.':
                excluded_region_indices = np.where(np.isin(baseline_data['region'], WORLD_REGIONS['USA']))[0]
            elif r[:3] == 'CN.':
                excluded_region_indices = np.where(np.isin(baseline_data['region'], WORLD_REGIONS['CHN']))[0]
        rs_to = np.where(~np.isin(baseline_data['index_region'][:], excluded_region_indices))[0]
        exports.loc[r] = np.ma.filled(flows[rs_from, :][:, rs_to], 0).sum()
    if _aggregate_chn_usa:
        for r in ['USA', 'CHN']:
            region_idx = np.where(np.isin(baseline_data['region'][:], WORLD_REGIONS[r]))[0]
            rs_from = np.where(np.isin(baseline_data['index_region'][:], region_idx) & (np.isin(baseline_data['index_sector'], sec_indices)))[0]
            rs_to = np.where(~np.isin(baseline_data['index_region'][:], region_idx))[0]
            exports.loc[r] = np.ma.filled(flows[rs_from, :][:, rs_to], 0).sum()
        exports = exports.loc[[i for i in exports.index if i[:3] not in ['CN.', 'US.']]]
    exports['share'] = exports['export'] / exports['export'].sum()
    exports = exports.sort_values(by='export', ascending=False)
    exports['cum_share'] = 0
    for i in range(len(exports)):
        exports.loc[exports.index[i], 'cum_share'] = float(exports.iloc[:i + 1]['share'].sum())
    return exports


def calc_trade_with_region(_regions=None, _sectors=None, _exclude_regions=None,
                           _baseline_path="/mnt/cluster_p/projects/acclimate/data/eora/Eora26-v199.82-2015-CHN-USA_naics_disagg.nc"):
    baseline_data = Dataset(_baseline_path)
    flows = baseline_data['flows'][:]
    res = pd.DataFrame(columns=['import', 'export', 'total'])
    target_region_indices = np.where(np.isin(baseline_data['region'][:], _regions))[0]
    if _sectors is None:
        _sectors = baseline_data['sector'][:]
    target_sector_indices = np.where(np.isin(baseline_data['sector'][:], _sectors))[0]
    target_rs_indices = np.where((np.isin(baseline_data['index_region'], target_region_indices)) & (np.isin(baseline_data['index_sector'], target_sector_indices)))[0]
    for r in tqdm.tqdm(baseline_data['region'][:]):
        if r not in _regions and r not in _exclude_regions:
            r_idx = np.where(baseline_data['region'][:] == r)[0][0]
            rs_indices = np.where((baseline_data['index_region'] == r_idx) & (np.isin(baseline_data['index_sector'], target_sector_indices)))[0]
            imp = np.ma.filled(flows[target_rs_indices, :][:, rs_indices], 0).sum()
            exp = np.ma.filled(flows[rs_indices, :][:, target_rs_indices], 0).sum()
            total = imp + exp
            res.loc[r] = [imp, exp, total]
    for col in ['import', 'export', 'total']:
        res[col + '_share'] = res[col] / res[col].sum()
    res = res.sort_values(by='total_share', ascending=False)
    res['cum_total_share'] = 0
    for i in range(len(res)):
        res.loc[res.index[i], 'cum_total_share'] = res.iloc[:i + 1]['total_share'].sum()
    return res
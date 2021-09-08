import copy
import itertools
import os
import pickle
import numpy as np
import yaml
import sys

from scipy import signal

sys.path.append('/home/robinmid/repos/acclimate/postproc/lib/acclimate')
sys.path.append('/home/robin/repos/postproc/lib/acclimate/')
import netcdf
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import us

WORLD_REGIONS = netcdf.world_regions
if 'NAM' in WORLD_REGIONS:
    WORLD_REGIONS['NMA'] = WORLD_REGIONS.pop('NAM')
if 'WORLD' not in WORLD_REGIONS:
    WORLD_REGIONS['WORLD'] = list(set([ctry for wr in netcdf.world_regions.values() for ctry in wr]))
if 'ROW' not in WORLD_REGIONS:
    WORLD_REGIONS['ROW'] = list(set(WORLD_REGIONS['WORLD']) - {'USA'} - set(WORLD_REGIONS['USA']))
if 'USA_REST_SANDY' not in WORLD_REGIONS:
    WORLD_REGIONS['USA_REST_SANDY'] = list(set(WORLD_REGIONS['USA']) - {'USA', 'US.NJ', 'US.NY'})
if 'USA_REST_HARVEY' not in WORLD_REGIONS:
    WORLD_REGIONS['USA_REST_HARVEY'] = list(set(WORLD_REGIONS['USA']) - {'USA', 'US.TX'})

SECTOR_GROUPS = {
    'ALLSECTORS': [
        'FCON',
        'AGRI',
        'FISH',
        'MINQ',
        'FOOD',
        'TEXL',
        'WOOD',
        'OILC',
        'METL',
        'MACH',
        'TREQ',
        'MANU',
        'RECY',
        'ELWA',
        'CONS',
        'REPA',
        'WHOT',
        'RETT',
        'GAST',
        'TRAN',
        'COMM',
        'FINC',
        'ADMI',
        'EDHE',
        'HOUS',
        'OTHE',
        'REXI'
    ],
    'PRIVSECTORS': [
        'AGRI',
        'FISH',
        'MINQ',
        'FOOD',
        'TEXL',
        'WOOD',
        'OILC',
        'METL',
        'MACH',
        'TREQ',
        'MANU',
        'RECY',
        'ELWA',
        'CONS',
        'REPA',
        'WHOT',
        'RETT',
        'GAST',
        'TRAN',
        'COMM',
        'FINC',
        'ADMI',
        'EDHE',
        'HOUS',
        'OTHE',
        'REXI'
    ]
}

state_gdp_path = "~/repos/harvey_scaling/data/external/CAGDP2__ALL_AREAS_2001_2019.csv"


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
    index = pd.MultiIndex.from_product([_data.get_regions(), _data.get_sectors(), _data.get_lambda_axis(),
                                        _data.get_duration_axis()])
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
    us_states_mapping = us.states.mapping('name', 'abbr')
    gdp_by_state = pd.read_csv(state_gdp_path)
    gdp_by_state.drop(gdp_by_state.index[-4:], inplace=True)
    gdp_by_state = gdp_by_state[gdp_by_state['Description'] == 'All industry total']
    gdp_by_state = gdp_by_state[gdp_by_state['GeoName'].isin(us_states_mapping.keys())]
    gdp_by_state['GeoName'] = gdp_by_state['GeoName'].apply(lambda x: 'US.' + us_states_mapping[x])
    gdp_by_state = gdp_by_state[gdp_by_state['GeoName'].isin(_states)]
    gdp_by_state = gdp_by_state[['GeoName', str(_year)]].astype({'GeoName': str, str(_year): int})
    gdp_by_state.set_index('GeoName', inplace=True)
    total_gdp = gdp_by_state[str(_year)].sum()
    for (scaled_l, scaled_d) in list(itertools.product(_data.get_lambda_axis(), _data.get_duration_axis())):
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
    for (scaled_l, scaled_d) in list(itertools.product(_data.get_lambda_axis(), _data.get_duration_axis())):
        direct_loss_iter = direct_loss.get_lambdavals(scaled_l).get_durationvals(scaled_d).get_data().sum()
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

# probably not needed anymore
# def get_aggregated_var_over_area(_regions, _var, _data):
#     if isinstance(_data, str):
#         dataset = Dataset(_data, "r", format="NETCDF4")
#     elif isinstance(_data, Dataset):
#         dataset = _data
#     else:
#         raise TypeError('Variable of type string or Dataset required.')
#     data = dataset["/agents/" + _var.replace('/', '')][:].data
#     region_indices = get_index_list(_regions, dataset["/region"][:])
#     variable_cumulative = 0
#     for sector_index in range(len(dataset.dimensions.get("sector"))):
#         variable_sector = np.nansum(data[:, sector_index, region_indices])
#         variable_cumulative = variable_cumulative + variable_sector
#     return variable_cumulative
#
#
# def get_var_over_ld_plot_for_area(_regions, _var, _lambda_axis, _duration_axis, _ensemble_meta, _experiment_series_dir):
#     result = np.zeros((len(_lambda_axis), len(_duration_axis)))
#     for l in range(len(_lambda_axis)):
#         for d in range(len(_duration_axis)):
#             iteration_path = os.path.join(_experiment_series_dir, _ensemble_meta[(_lambda_axis[l], _duration_axis[d])][
#                 'iter_name'] + "/output.nc")
#             try:
#                 #                iteration_dataset = dataset_dict[(lambda_axis[l], duration_axis[d])]
#                 result[l, d] = get_aggregated_var_over_area(_regions, _var, iteration_path)
#                 #                result[l, d] = get_aggregated_var_over_area_from_dataset(regions, var, iteration_dataset)
#                 print("l = {}, d = {}, val = {} | {}".format(_lambda_axis[l], _duration_axis[d], result[l, d],
#                                                              iteration_path))
#             #                print("l = {}, d = {}, val = {}".format(lambda_axis[l], duration_axis[d], result[l, d]))
#             except Exception as e:
#                 print("l = {}, d = {} from file {} could not be loaded.".format(_lambda_axis[l], _duration_axis[d],
#                                                                                 iteration_path))
#                 print(e)
#     return result

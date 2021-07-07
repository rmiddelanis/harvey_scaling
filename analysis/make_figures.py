import copy
import json
import string

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import os
import sys
sys.path.append("../")
from data.calc_initial_forcing_intensity_HARVEY import plot_polygon, load_hwm, alpha_shape, alpha
from analysis.dataformat import AggrData
from analysis.utils import WORLD_REGIONS
from scipy.interpolate import interp1d, RectBivariateSpline
import matplotlib as mpl
from scipy.ndimage import gaussian_filter, zoom

rootdir = os.path.join(os.getenv("HOME"), 'repos/harvey_scaling/')

mpl.rcdefaults()

MAX_COLUMN_HEIGHT = 9.60
MAX_FIG_WIDTH_WIDE = 7.07
MAX_FIG_WIDTH_NARROW = 3.45
FSIZE_TINY = 6
FSIZE_SMALL = 8
FSIZE_MEDIUM = 10
FSIZE_LARGE = 12

plt.rc('font', size=FSIZE_SMALL)  # controls default text sizes
plt.rc('axes', titlesize=FSIZE_SMALL)  # fontsize of the axes title
plt.rc('axes', labelsize=FSIZE_SMALL)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=FSIZE_SMALL)  # fontsize of the tick labels
plt.rc('ytick', labelsize=FSIZE_SMALL)  # fontsize of the tick labels
plt.rc('legend', fontsize=FSIZE_SMALL)  # legend fontsize
plt.rc('figure', titlesize=FSIZE_LARGE)  # fontsize of the figure title
plt.rc('axes', linewidth=0.5)  # fontsize of the figure title


def plot_initial_claims(_plot_from='07-01-2017', _plot_to='01-01-2018', _shade_from='2017-08-26',
                        _shade_to='2017-10-28', _outfile=None):
    data = pd.read_csv(os.path.join(rootdir, "data/external/TXICLAIMS_1986_to_2019.csv"), na_values=['.'])
    data.dropna(inplace=True)
    data.DATE = pd.to_datetime(data.DATE)
    data.TXICLAIMS = data.TXICLAIMS.astype(int)
    fig, ax = plt.subplots(figsize=(MAX_FIG_WIDTH_NARROW, MAX_FIG_WIDTH_NARROW))
    data = data[(data.DATE >= _plot_from) & (data.DATE <= _plot_to)]
    data.plot(x='DATE', y='TXICLAIMS', ax=ax, label='TX Initial Claims', xlabel='time', ylabel='claims')
    shade_data = data[(data.DATE >= _shade_from) & (data.DATE <= _shade_to)]
    shade_x = shade_data.DATE.values
    shade_y1 = shade_data.TXICLAIMS.values
    shade_y2 = max(data[data.DATE == _shade_from].TXICLAIMS.iloc[0], data[data.DATE == _shade_to].TXICLAIMS.iloc[0])
    plt.fill_between(shade_x, shade_y1, shade_y2, alpha=0.3, label='estimated Harvey\neffect')
    plt.legend() #(loc='upper right')
    plt.tight_layout()
    if _outfile is not None:
        plt.savefig(_outfile, dpi=300)
    plt.show()


def plot_radius_extension_map(_numbering=None, _outfile=None):
    states = ['Louisiana', 'Texas']
    affected_counties = json.load(open(os.path.join(rootdir, 'data/generated/affected_counties.json'), 'rb'))
    for key in list(affected_counties.keys()):
        affected_counties[int(key)] = affected_counties.pop(key)
    radius_extensions = [int(re) for re in affected_counties.keys()]
    fig_width = MAX_FIG_WIDTH_NARROW
    fig_height = fig_width * 0.67
    fig, (ax1, cbar_ax) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    cbar_width = 0.02
    cbar_x2 = 1 - (0.15 * MAX_FIG_WIDTH_NARROW / fig_width)
    cbar_x1 = cbar_x2 - cbar_width
    cbar_y1 = 0
    cbar_y2 = 1
    cbar_dist = 0.02
    cbar_ax.set_position([cbar_x1, cbar_y1, cbar_x2 - cbar_x1, cbar_y2 - cbar_y1])
    ax1_bbox_x1 = 0
    ax1_bbox_x2 = cbar_x1 - cbar_dist
    ax1_bbox_y1 = 0
    ax1_bbox_y2 = 1
    ax1.set_position([ax1_bbox_x1, ax1_bbox_y1, ax1_bbox_x2 - ax1_bbox_x1, ax1_bbox_y2 - ax1_bbox_y1])
    ax1.axis('off')
    cmap = plt.cm.get_cmap('Reds', lut=len(radius_extensions))
    bounds = np.arange(len(radius_extensions) + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb2 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                    norm=norm,
                                    # to use 'extend', you must
                                    # specify two extra boundaries:
                                    boundaries=[0] + bounds,
                                    ticks=bounds,  # optional
                                    spacing='proportional',
                                    orientation='vertical')
    cb2.set_label('radius extension [km]')
    cb2.set_ticks(np.arange(0.5, len(radius_extensions), 1))
    cb2.set_ticklabels([str(int(r / 1e3)) for r in radius_extensions])
    hwm_gdf = load_hwm()
    concave_hull, _ = alpha_shape(hwm_gdf.geometry, alpha=alpha)
    us_states_shp = gpd.read_file(os.path.join(rootdir, 'data/external/gadm36_USA_shp/gadm36_USA_1.shp')).to_crs(epsg=3663)
    us_county_shp = gpd.read_file(os.path.join(rootdir, 'data/external/gadm36_USA_shp/gadm36_USA_2.shp')).to_crs(epsg=3663)
    us_county_shp[(us_county_shp['NAME_1'].isin(states)) & (~us_county_shp['HASC_2'].isin(np.concatenate(list(affected_counties.values()))))].plot(ax=ax1, facecolor='lightgrey')
    us_states_shp[us_states_shp['NAME_1'].isin(states)].plot(ax=ax1, edgecolor='k', facecolor='none', linewidth=1)
    for ax_idx, radius_extension in enumerate(radius_extensions):
        plot_polygon(concave_hull.buffer(radius_extension), ax1, _fc='none', _ec=cmap(ax_idx))
        us_county_shp[us_county_shp['HASC_2'].isin(affected_counties[radius_extension])].plot(ax=ax1, color=cmap(ax_idx))
    ax1.set_xticks([])
    ax1.set_yticks([])
    hwm_gdf.plot(ax=ax1, markersize=0.4, color='midnightblue', marker='o')
    ylim = (2.55e6, 3.55e6)
    xlim = (7e5, 1.85e6)
    metric = ylim[1] - ylim[0]
    x0, y0 = xlim[0], ylim[0]
    y_pos = 0.15 * metric
    y_radius = 0.03 * metric
    y_distance = 0.05 * metric
    x_pos = 0.6 * metric
    x_radius = 0.06 * metric
    x_distance = 0.01 * metric
    ax1.add_patch(mpl.patches.Ellipse((x0 + x_pos, y0 + y_pos), 0.5 * y_radius, 0.5 * y_radius, color='midnightblue'))
    ax1.text(x0 + (x_pos + x_radius + x_distance), y0 + y_pos, 'high water marks', va='center', ha='left')
    y_pos = y_pos - y_distance
    num_lines = 5
    for line_idx in range(num_lines):
        x_offset = (line_idx - (num_lines - 1) / 2) * (x_radius / (num_lines - 1))
        x_vals = [x0 + (x_pos + x_offset), x0 + (x_pos + x_offset)]
        y_vals = [y0 + (y_pos - y_radius / 2), y0 + (y_pos + y_radius / 2)]
        cmap_val = 0.1 + line_idx / (num_lines - 1) * 0.8
        ax1.plot(x_vals, y_vals, color=cmap(cmap_val), transform=ax1.transData, alpha=0.3)
    ax1.text(x0 + (x_pos + x_radius + x_distance), y0 + y_pos, 'radius envelopes', va='center',
             ha='left', transform=ax1.transData)
    y_pos = y_pos - y_distance
    corner_ll = np.array([x0 + (x_pos - x_radius / 2), y0 + (y_pos - y_radius / 2)])
    corner_ul = np.array([x0 + (x_pos - x_radius / 2), y0 + (y_pos + y_radius / 2)])
    corner_ur = np.array([x0 + (x_pos + x_radius / 2), y0 + (y_pos + y_radius / 2)])
    corner_lr = np.array([x0 + (x_pos + x_radius / 2), y0 + (y_pos - y_radius / 2)])
    triangle_1 = mpl.patches.Polygon(np.array([corner_ll, corner_ul, corner_ur * [0.99, 1]]), color=cmap(0.3),
                                     transform=ax1.transData)
    triangle_2 = mpl.patches.Polygon(np.array([corner_ll * [1.01, 1], corner_ur, corner_lr]), color=cmap(0.6),
                                     transform=ax1.transData)
    ax1.add_patch(triangle_1)
    ax1.add_patch(triangle_2)
    ax1.text(x0 + x_pos + (x_radius + x_distance), y0 + y_pos, 'affected counties', va='center',
             ha='left', transform=ax1.transData)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.axis('off')
    if _numbering is not None:
        fig.text(0, 1, _numbering, ha='left', va='top', fontweight='bold')
    if _outfile is not None:
        plt.savefig(_outfile, dpi=300, format='pdf')
    plt.show()


def plot_radius_extension_impact(_outfile=None):
    fig, ax = plt.subplots(figsize=(MAX_FIG_WIDTH_NARROW, 0.75 * MAX_FIG_WIDTH_NARROW))
    initial_forcing_intensities = json.load(open(os.path.join(rootdir, "data/generated/initial_forcing_params.json"), "rb"))
    affected_counties = json.load(open(os.path.join(rootdir, 'data/generated/affected_counties.json'), 'rb'))
    for key in list(affected_counties.keys()):
        affected_counties[int(key)] = affected_counties.pop(key)
    re = [int(re) for re in affected_counties.keys()]
    for _s in ['LA', 'TX']:
        m_f0_i = initial_forcing_intensities['params'][_s]['m']
        c_f0_i = initial_forcing_intensities['params'][_s]['c']
        ax.plot([re[0], re[-1]], [m_f0_i * re[0] + c_f0_i, m_f0_i * re[-1] + c_f0_i], '--k')
        if _s == 'LA':
            y_pos = 0
            va = 'bottom'
        elif _s == 'TX':
            y_pos = 0.75
            va = 'top'
        ax.text(1, y_pos, "y={0:1.3e}x+{1:1.4f}".format(m_f0_i * 1e3, c_f0_i), ha='right', va=va, transform=ax.transAxes)
        # ax1.scatter(x, [initial_forcing_intensities['points'][re][_s] for re in radius_extensions], label=_s, s=6)
        ax.plot(re, [initial_forcing_intensities['points'][str(re)][_s] for re in re], label=_s)
    ax.set_yticks([0.1 * i for i in range(6)])
    # ax1.set_yticklabels(np.arange(0, 0.55, 0.1))
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, borderpad=0., handletextpad=0.5, handlelength=1)
    ax.set_xlabel('Radius extension [km]')
    plt.tight_layout()
    ax.set_position([0.18, ax.get_position().y0, ax.get_position().width - (0.18 - ax.get_position().x0),
                     ax.get_position().height])
    fig.text(-0.22, 0.5, r'$f^{(0)}_s$', fontsize=FSIZE_MEDIUM, ha='left', va='center', transform=ax.transAxes)
    ax.set_xticks(re)
    ax.set_xticklabels(["{}".format(int(_re / 1e3)) for _re in re], rotation=90)
    plt.show()
    if isinstance(_outfile, str):
        fig.savefig(_outfile, dpi=300)


def prepare_heatmap_figure(_data: AggrData, _type: str, _x_ax: bool, _scale_factor=1.0, gmt_anomaly_0=1.0,
                           _sst_gmt_factor=1.0, _numbering=None):
    fig_width = MAX_FIG_WIDTH_WIDE * 0.8 * _scale_factor
    if _x_ax:
        fig_height = MAX_FIG_WIDTH_WIDE * _scale_factor * 0.5
        ax_bbox_y1 = 0.2 / _scale_factor
    else:
        fig_height = MAX_FIG_WIDTH_WIDE * _scale_factor * 0.45
        ax_bbox_y1 = 0.05 / _scale_factor
    ax_bbox_x1 = 0.15 / _scale_factor
    ax_bbox_x2 = 1 - 0.17 / _scale_factor
    ax_width = ax_bbox_x2 - ax_bbox_x1
    dist_ax_cbar = 0.02
    cbar_width = 0.02
    cbar_bbox_x1 = ax_bbox_x1 + ax_width + dist_ax_cbar
    ax_height = 0.99 - ax_bbox_y1
    ax_bbox = (ax_bbox_x1, ax_bbox_y1, ax_width, ax_height)
    cbar_bbox = (cbar_bbox_x1, ax_bbox_y1, cbar_width, ax_height)
    print(ax_bbox, "\n", cbar_bbox)
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax, cbar_ax = fig.add_axes(ax_bbox), fig.add_axes(cbar_bbox)
    if _type == 'heatmap_cut':
        cbar_ax.remove()
    if _type == 'heatmap':
        ax.tick_params(axis='y', labelrotation=0)
        ax.set_ylim(-0.5, len(_data.get_lambda_axis()) - 0.5)
        ax.set_yticks(np.arange(0, len(_data.get_lambda_axis()), 1))
        ax.set_yticklabels([int(l) for l in _data.get_lambda_axis() / 1e3])
        ax.set_ylabel('radius change (km)')
    if _x_ax:
        ax.set_xlim(-0.5, len(_data.get_duration_axis()) - 0.5)
        ax.set_xlim(-0.5, len(_data.get_duration_axis()) - 0.5)
        ax.set_xticks(np.arange(0, len(_data.get_duration_axis()), 1))
        ax.set_xticklabels(["{0:1.2f}".format(gmt_anomaly_0 + d / _sst_gmt_factor) for d in _data.get_duration_axis()])
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xlabel('global mean temperature anomaly\n$(\degree C)$')
    else:
        ax.set_xlabel('')
        ax.set_xticklabels([])
        # ax.set_xlim(0, len(_data.get_duration_axis()) - 0.5)
        ax.set_xticks(np.arange(0, len(_data.get_duration_axis()), 1))
    if _numbering is not None:
        fig.text(0, 1, _numbering, ha='left', va='top', fontweight='bold')
    return fig, ax, cbar_ax


def make_heatmap(_data: AggrData, _agg_method: str, base_point=None, contour_distance=None, _gauss_filter=True,
                 _gauss_sigma=1, _gauss_truncate=1, _outfile=None, _slopes=None, _plot_cuts=False, _label=None,
                 _sst_gmt_factor=1.0, **kwargs):
    print('first', _sst_gmt_factor)
    if _data.shape()[0] != 1 or _data.shape()[1] != 1 or _data.shape()[2] != 1:
        raise ValueError("All dimensions of the dataset except lambda and duration and time must be 1.")
    fig, ax, cbar_ax = prepare_heatmap_figure(_data, _type='heatmap', _x_ax=(not _plot_cuts),
                                              _sst_gmt_factor=_sst_gmt_factor, _numbering='a')
    _data_aggregated = copy.deepcopy(_data.clip(1))
    if _agg_method == 'sum':
        _data_aggregated.data_capsule.data = _data.get_data().sum(axis=-1, keepdims=True)
    elif _agg_method == 'min':
        _data_aggregated.data_capsule.data = _data.get_data().min(axis=-1, keepdims=True)
    elif _agg_method == 'max':
        _data_aggregated.data_capsule.data = _data.get_data().max(axis=-1, keepdims=True)
    data_array = _data_aggregated.get_data().reshape((len(_data.get_lambda_axis()), len(_data.get_duration_axis())))
    data_filtered = gaussian_filter(data_array, sigma=_gauss_sigma, mode='nearest', truncate=_gauss_truncate)
    if _gauss_filter:
        plot_data = data_filtered
    else:
        plot_data = data_array
    im = ax.imshow(plot_data, origin='lower', aspect='auto')
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.set_label(_label)
    if contour_distance is not None:
        contour_data = zoom(data_filtered, 3)
        vmax = data_array.max()
        vmin = data_array.min()
        levels = np.arange(vmin, vmax, contour_distance)
        contours = ax.contour(contour_data, levels, colors='k', origin='lower',
                              extent=(-0.5, plot_data.shape[1] - 0.5, -0.5, plot_data.shape[0] - 0.5))
    if base_point is None:
        duration_0, lambda_0 = 0, 0  # base point duration is in degC CHANGE compared to the unscaled hurricane, not in degC global temperature anomaly
    else:
        duration_0, lambda_0 = base_point
    base_x = np.where(_data_aggregated.get_duration_axis() == duration_0)[0][0]
    base_y = np.where(_data_aggregated.get_lambda_axis() == lambda_0)[0][0]
    base_z = _data_aggregated.get_durationvals(duration_0).get_lambdavals(lambda_0).get_data().flatten()[0]
    if _slopes is not None:
        for idx, slope in enumerate(sorted(_slopes, reverse=True)):
            d_i = (_data_aggregated.get_lambda_axis()[-1] - _data_aggregated.get_lambda_axis()[0]) / len(
                _data_aggregated.get_lambda_axis())
            d_d = (_data_aggregated.get_duration_axis()[-1] - _data_aggregated.get_duration_axis()[0]) / len(
                _data_aggregated.get_duration_axis())
            m = slope * d_d / d_i / _sst_gmt_factor  # slopes are in km/degC GMT change -> translate into km/degC SST change
            b = base_y - m * base_x
            x_max = min(len(_data_aggregated.get_duration_axis()) - 1,
                        ((len(_data_aggregated.get_lambda_axis()) - 1) - b) / m)
            y_max = m * x_max + b
            ax.plot([base_x, x_max], [m * base_x + b, y_max], c='w')
            ax.text(base_x + (x_max - base_x) / 2, m * (base_x + (x_max - base_x) / 2) + b, string.ascii_uppercase[idx],
                    c='w')
    ax.plot(base_x, base_y, marker='x', markersize=8, color='k')
    ax.text(base_x, base_y, '\n({0:1.3f})'.format(base_z), color='k', fontsize=FSIZE_SMALL, ha='center', va='top')
    if _plot_cuts:
        make_heatmap_cut(_data_aggregated, _agg_method, _slopes, _gauss_filter=_gauss_filter, _gauss_sigma=_gauss_sigma,
                         _plot_xax=True, _gauss_truncate=_gauss_truncate, _label=_label, _duration_0=duration_0,
                         _sst_gmt_factor=_sst_gmt_factor, **kwargs)
    plt.tight_layout()
    if _outfile is not None:
        fig.savefig(_outfile, dpi=300)
    plt.show()


def make_heatmap_cut(_data: AggrData, _agg_method: str, _slopes, _outfile=None, _gauss_filter=True, _gauss_sigma=1,
                     _gauss_truncate=1, _plot_xax=True, _show_baseline=False, _label=None, _sst_gmt_factor=1.0,
                     _duration_0=0):
    if _data.shape()[0] != 1 or _data.shape()[1] != 1 or _data.shape()[2] != 1:
        raise ValueError("All dimensions of the dataset except lambda and duration and time must be 1.")

    fig, ax, _ = prepare_heatmap_figure(_data, _type='heatmap_cut', _x_ax=_plot_xax, _sst_gmt_factor=_sst_gmt_factor,
                                        _numbering='b')
    if _agg_method == 'sum':
        data_array = _data.get_data().sum(axis=-1, keepdims=True)
    elif _agg_method == 'min':
        data_array = _data.get_data().min(axis=-1, keepdims=True)
    elif _agg_method == 'max':
        data_array = _data.get_data().max(axis=-1, keepdims=True)
    data_array = data_array.reshape((len(_data.get_lambda_axis()), len(_data.get_duration_axis())))
    if _gauss_filter:
        data_array = gaussian_filter(data_array, sigma=_gauss_sigma, mode='nearest', truncate=_gauss_truncate)
    interp = RectBivariateSpline(_data.get_lambda_axis(), _data.get_duration_axis(), data_array, s=0)
    for idx, slope in enumerate(sorted(_slopes, reverse=True)):
        durations = _data.get_duration_axis()[_data.get_duration_axis() >= _duration_0]
        x_offset = len(_data.get_duration_axis()[_data.get_duration_axis() < _duration_0])
        lambdas = durations * slope / _sst_gmt_factor  # slopes are in km/degC GMT change. Therefore, translate into km/degC SST change:
        lambdas = lambdas[lambdas <= _data.get_lambda_axis()[-1]]
        durations = durations[:len(lambdas)]
        z = [interp(lambdas[i], durations[i])[0, 0] for i in range(len(durations))]
        ax.plot(np.arange(len(durations)) + x_offset, z,
                label=string.ascii_uppercase[idx] + ": m = {0:1.0f} km / $\degree C$".format(_slopes[idx] / 1e3))
    if _show_baseline:
        if (_data.get_data()[..., 0].flatten() != _data.get_data()[..., 0].flatten()[0]).sum() == 0:
            if _agg_method == 'sum':
                factor = _data.get_sim_duration()
            elif _agg_method in ['min', 'max']:
                factor = 1
            if _show_baseline:
                ax.axhline(y=_data.get_data()[..., 0].flatten()[0] * factor, linestyle='--', label='baseline')
        else:
            print('Warning. Ambiguous baseline values. Baseline not shown')
    ax.set_ylabel(_label)
    ax.legend()
    plt.show()


def make_region_impact_plot(_data: AggrData, _absolute_consumption_deviation_thres=1e8, _include_row=True,
                            _filter_on_last_scaled_scenario=False, _comparison_simulations=None, _absolute_unit=1e9,
                            _numbering=True, _include_usa=False, _break_absolute_axis=(-14, -1.3),
                            _scenario_labels=None, _region_selection=None, _use_absolute_deviation=True):
    if _data.get_sim_duration() != 1:
        raise ValueError('Must pass data with exactly one timestep')
    if 'consumption_baseline' not in _data.get_vars() or sum(
            ['consumption_deviation_' in s for s in _data.get_vars()]) != 1:
        raise ValueError('Must pass data with variable \'consumption_baseline\' and exactly one consumption deviation '
                         'variable')
    if len(_data.get_regions()) <= 1:
        raise ValueError('Must pass data with at least two regions')
    if len(_data.get_sectors()) != 1:
        raise ValueError('Must pass data with only one sector')

    if _comparison_simulations is None:
        _comparison_simulations = [(-1, -1)]

    if _comparison_simulations[0] != (0, 0):
        _comparison_simulations = [(0, 0)] + _comparison_simulations
    num_scenarios = len(_comparison_simulations)
    deviation_var = 'consumption_deviation_365d'
    duration_axis = _data.get_duration_axis()
    lambda_axis = _data.get_lambda_axis()
    deviation_df = pd.DataFrame()
    deviation_df['region_name'] = _data.get_regions()
    deviation_df.set_index('region_name', inplace=True, drop=False)
    deviation_df['baseline_consumption'] = _data.get_vars('consumption_baseline').get_data()[
                                               ..., 0, 0, 0].flatten() * 1e3
    for scenario_idx, _comparison_simulation in enumerate(_comparison_simulations):
        scaled_l = _comparison_simulation[0]
        scaled_d = _comparison_simulation[1]
        if scaled_d == -1:
            scaled_d = duration_axis[-1]
        if scaled_l == -1:
            scaled_l = lambda_axis[-1]
        _comparison_simulations[scenario_idx] = (scaled_l, scaled_d)
        relative_dev_percent = _data.get_vars(deviation_var).get_lambdavals(
            scaled_l).get_durationvals(scaled_d).get_data().flatten()
        deviation_df['relative_deviation_{}'.format(scenario_idx)] = relative_dev_percent
        deviation_df['absolute_deviation_{}'.format(scenario_idx)] = relative_dev_percent / 100 \
                                                                     * deviation_df['baseline_consumption'] * 365
        deviation_df['absolute_deviation_{}_noSgn'.format(scenario_idx)] = deviation_df[
            'absolute_deviation_{}'.format(scenario_idx)].apply(lambda x: abs(x))
        deviation_df['deviation_quotient_{}'.format(scenario_idx)] = deviation_df[
                                                                         'absolute_deviation_{}'.format(scenario_idx)] / \
                                                                     deviation_df['absolute_deviation_0']
    if _filter_on_last_scaled_scenario:
        filter_var = 'absolute_deviation_{}_noSgn'.format(scenario_idx)
    else:
        filter_var = 'absolute_deviation_0_noSgn'
    deviation_selection = deviation_df[(deviation_df[filter_var] >= _absolute_consumption_deviation_thres) &
                                       (~deviation_df.index.isin(set(WORLD_REGIONS['USA']) - {'USA'})) &
                                       (~deviation_df.index.isin(set(WORLD_REGIONS['CHN']) - {'CHN'}))
                                       ]
    if _region_selection is None:
        deviation_selection = deviation_selection[
            (~deviation_selection.index.isin(set(WORLD_REGIONS.keys()) - {'CHN', 'USA'}))]
    else:
        continents = ['EU28', 'ASI', 'LAM', 'WORLD', 'AFR', 'OCE']
        if _region_selection == 'continents':
            selected_regions = continents
        elif _region_selection == 'all':
            selected_regions = list(
                set(deviation_selection.index) - (set(WORLD_REGIONS.keys()) - set(continents) - {'CHN'}))
        if _include_usa:
            selected_regions.append('USA')
        deviation_selection = deviation_selection[(deviation_selection.index.isin(selected_regions))]
    if _include_row and _region_selection is False:
        rest_of_world = deviation_df[(~deviation_df.index.isin(deviation_selection.index)) &
                                     (~deviation_df.index.isin(set(WORLD_REGIONS.keys()) - {'CHN', 'USA'})) &
                                     (~deviation_df.index.isin(set(WORLD_REGIONS['USA']) - {'USA'})) &
                                     (~deviation_df.index.isin(set(WORLD_REGIONS['CHN']) - {'CHN'}))
                                     ]
        rest_of_world = rest_of_world.sum()
        for scenario_idx in range(num_scenarios):
            rest_of_world['relative_deviation_{}'] = rest_of_world['absolute_deviation_{}'] / rest_of_world[
                'baseline_consumption']
            rest_of_world['absolute_deviation_{}_noSgn'] = abs(rest_of_world['absolute_deviation_0'])
            rest_of_world['deviation_quotient_{}'] = rest_of_world['relative_deviation_{}'] / rest_of_world[
                'relative_deviation_0']
        rest_of_world['region_name'] = 'ROW'
        deviation_selection.loc['ROW'] = rest_of_world
    deviation_selection = deviation_selection.sort_values(by=filter_var, ascending=False)
    selected_world_regions = list(
        deviation_selection.index[deviation_selection.index.isin(set(WORLD_REGIONS.keys()) - {'CHN', 'USA'})])
    selected_countries = list(
        deviation_selection.index[~deviation_selection.index.isin(set(WORLD_REGIONS.keys()) - {'CHN', 'USA'})])
    deviation_selection = deviation_selection.loc[selected_world_regions + selected_countries]
    if not _include_usa:
        deviation_selection = deviation_selection.drop('USA')
    labels = selected_world_regions + selected_countries
    deviations = []
    deviation_quotients = []
    for scenario_idx in range(num_scenarios):
        if _use_absolute_deviation:
            deviation = list(deviation_selection['absolute_deviation_{}'.format(scenario_idx)] / _absolute_unit)
        else:
            deviation = list(deviation_selection['relative_deviation_{}'.format(scenario_idx)])
        deviation_quotient = list(deviation_selection['deviation_quotient_{}'.format(scenario_idx)])
        if len(selected_world_regions) > 0:
            deviation = deviation[:len(selected_world_regions)] + [0] + deviation[len(selected_world_regions):]
            deviation_quotient = deviation_quotient[:len(selected_world_regions)] + [0] + deviation_quotient[
                                                                                          len(selected_world_regions):]
        deviations.append(deviation)
        deviation_quotients.append(deviation_quotient)
    x = np.arange(len(labels) + 1 if len(selected_world_regions) > 0 else 0)  # the label locations
    width = 0.8  # the width of the bars
    bar_width = width / num_scenarios
    height_ratios = [0.2, 0.05, 0.25] if _break_absolute_axis is not False else [0.25, 0, 0.25]
    fig = plt.figure(figsize=(MAX_FIG_WIDTH_WIDE, MAX_FIG_WIDTH_NARROW * 1.5))
    gs = fig.add_gridspec(3, 1, height_ratios=height_ratios)
    ax2 = fig.add_subplot(gs[2])
    ax1_1 = fig.add_subplot(gs[0], sharex=ax2)
    ax1_2 = fig.add_subplot(gs[1], sharex=ax2)
    axs = [ax1_1, ax1_2, ax2]
    cmap = plt.cm.get_cmap('Blues')
    colors = [cmap(0.2 + i * (0.6 / (num_scenarios - 1))) for i in range(num_scenarios)]
    for scenario_idx in range(num_scenarios):
        deviation = deviations[scenario_idx]
        if scenario_idx == 0:
            label = 'unscaled'
        else:
            if _scenario_labels is None:
                label = 'scaled_{}'.format(scenario_idx)
            else:
                if len(_scenario_labels) == num_scenarios:
                    label = _scenario_labels[scenario_idx]
                elif len(_scenario_labels) == num_scenarios - 1:
                    label = _scenario_labels[scenario_idx]
        color = colors[scenario_idx]
        ax1_1.bar(x - width / 2 + scenario_idx * bar_width, deviation, bar_width, label=label, color=color)
        ax1_2.bar(x - width / 2 + scenario_idx * bar_width, deviation, bar_width, label=label, color=color)
    absolute_unit_dict = {
        1e9: 'bn',
        1e6: 'm',
        1e3: 't'
    }
    if _break_absolute_axis is not False:
        ax1_1.set_ylim(_break_absolute_axis[1], np.array(deviations).max() + 0.05)
        ax1_2.set_ylim(ax1_2.get_ylim()[0], _break_absolute_axis[0])
        ax1_1.spines['bottom'].set_visible(False)
        ax1_2.spines['top'].set_visible(False)
        ax1_1.tick_params(bottom=False, top=False)
        ax1_1.set_xticklabels([])
        ax1_2.tick_params(labeltop=False)  # don't put tick labels at the top
        ax1_2.xaxis.tick_bottom()
    ax1_1.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    for scenario_idx in range(1, num_scenarios):
        color = colors[scenario_idx]
        deviation_quotient = deviation_quotients[scenario_idx]
        ax2.bar(x - width / 2 + scenario_idx * bar_width, deviation_quotient, bar_width, color=color)
    for scenario_idx, (scaled_l, scaled_d) in enumerate(_comparison_simulations):
        if scenario_idx == num_scenarios - 1:
            color = colors[scenario_idx]
            scaled_f0 = _data.scaled_scenarios[(scaled_l, scaled_d)]['params']['mean_initial_forcing_intensity']
            unscaled_f0 = _data.scaled_scenarios[(0, 0)]['params']['mean_initial_forcing_intensity']
            forcing_ratio = scaled_f0 / unscaled_f0
            ax2.axhline(forcing_ratio, label='intial forcing ratio', linestyle='--', color=color)
            scaled_damage = _data.scaled_scenarios[(scaled_l, scaled_d)]['params']['damage']
            unscaled_damage = _data.scaled_scenarios[(0, 0)]['params']['damage']
            damage_ratio = scaled_damage / unscaled_damage
            ax2.axhline(damage_ratio, label='total damage ratio', linestyle='dotted', color=color)
    ax2.set_xticks(x[x != len(selected_world_regions)])
    ax2.set_xticklabels(labels, rotation=90)
    ax2.legend(frameon=False)
    plt.setp(ax1_1.get_xticklabels(), visible=False)
    plt.setp(ax1_2.get_xticklabels(), visible=False)
    if _break_absolute_axis is False:
        ax1_2.set_visible(False)
    plt.tight_layout()
    ax_x0 = 0.11
    for ax in axs:
        ax_pos = ax.get_position()
        ax.set_position([ax_x0, ax_pos.y0, ax_pos.x1 - ax_x0, ax_pos.height])
    if _numbering:
        for ax_idx, ax in enumerate([ax1_1, ax2]):
            fig.text(0, ax.get_position().y1, chr(ax_idx + 97), fontweight='bold', ha='left', va='center')
    upper_label_x = 0
    upper_label_y = ax1_2.get_position().y0 + (ax1_1.get_position().y1 - ax1_2.get_position().y0) / 2
    if _use_absolute_deviation:
        ax1_label_row1 = 'absolute deviation'
        ax1_label_row2 = '\n({} USD)'.format(absolute_unit_dict.get(_absolute_unit, _absolute_unit))
    else:
        ax1_label_row1 = 'relative deviation (%)'
        ax1_label_row2 = '\n'
    fig.text(upper_label_x, upper_label_y, ax1_label_row1, ha='left', va='center', rotation=90)
    fig.text(upper_label_x, upper_label_y, ax1_label_row2, ha='left', va='center', rotation=90)
    lower_label_x = 0
    lower_label_y = ax2.get_position().y0 + (ax2.get_position().y1 - ax2.get_position().y0) / 2
    fig.text(lower_label_x, lower_label_y, 'amplification ratio', ha='left', va='center', rotation=90)
    fig.text(lower_label_x, lower_label_y, '\n(scaled / unscaled)', ha='left', va='center', rotation=90)
    if _break_absolute_axis is not False:
        d = .01  # how big to make the diagonal lines in figure coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=fig.transFigure, color='k', clip_on=False)
        ax1_1_pos = ax1_1.get_position()
        ax1_1.plot((ax1_1_pos.x0 - d, ax1_1_pos.x0 + d), (ax1_1_pos.y0 - d, ax1_1_pos.y0 + d), **kwargs)
        ax1_1.plot((ax1_1_pos.x1 - d, ax1_1_pos.x1 + d), (ax1_1_pos.y0 - d, ax1_1_pos.y0 + d), **kwargs)
        ax1_2_pos = ax1_2.get_position()
        ax1_2.plot((ax1_2_pos.x0 - d, ax1_2_pos.x0 + d), (ax1_2_pos.y1 - d, ax1_2_pos.y1 + d), **kwargs)
        ax1_2.plot((ax1_2_pos.x1 - d, ax1_2_pos.x1 + d), (ax1_2_pos.y1 - d, ax1_2_pos.y1 + d), **kwargs)
    plt.show()


def make_scatter_plot(_data: AggrData, _x_dim='damage', _y_dim='var', _z_dim='radius', _var_name='', _z_vis='cbar',
                      _temperature_offset=1, _hide_every_other_ticklabel=False, _num_cbar_bins=None, _cmap='hot',
                      _plot_regressions=True, _var_scale=1, _numbering=None, _xlabels=True, _ylabels=True):
    if _data.get_sim_duration() != 1:
        raise ValueError('Must pass data with only one timestep')
    if len(_data.get_vars()) != 1:
        raise ValueError('Must pass data with only one variable')
    if len(_data.get_regions()) != 1:
        raise ValueError('Must pass data with only one region')
    if len(_data.get_sectors()) != 1:
        raise ValueError('Must pass data with only one sector')
    if _x_dim != 'var' and _y_dim != 'var' and _z_dim != 'var':
        raise ValueError('One dimension must be the data variable.')
    if _x_dim == _y_dim or _y_dim == _z_dim or _z_dim == _x_dim:
        raise ValueError('all dimensions must be different.')

    figsize = (MAX_FIG_WIDTH_NARROW, 0.9 * MAX_FIG_WIDTH_NARROW)

    data_xyz = [[], [], []]
    for _d_idx, _d in enumerate(_data.get_duration_axis()):
        for _l_idx, _l in enumerate(_data.get_lambda_axis()):
            temp = {'tr': _data.scaled_scenarios[(_l, _d)]['params']['tr'],
                    'tau': _data.scaled_scenarios[(_l, _d)]['params']['tau'],
                    'var': _data.get_lambdavals(_l).get_durationvals(_d).get_data().flatten()[0] / _var_scale,
                    're': int(_l / 1e3),
                    'clausius_clapeyron': _d + _temperature_offset,
                    'initial_forcing_TX': _data.scaled_scenarios[(_l, _d)]['params']['forcing']['TX']['intensity'],
                    'initial_forcing_LA': _data.scaled_scenarios[(_l, _d)]['params']['forcing']['LA']['intensity'],
                    'mean_initial_forcing': _data.scaled_scenarios[(_l, _d)]['params'][
                        'mean_initial_forcing_intensity'],
                    }
            for data_dim, dim in enumerate([_x_dim, _y_dim, _z_dim]):
                data_xyz[data_dim].append(temp[dim])
    data_xyz = np.array(data_xyz)

    labels = {#'damage': 'direct disaster damage\n(bn USD)',
              'radius': 'radius extension\n(km)',
              'clausius_clapeyron': 'global mean temperature\nanomaly (°C)',
              'var': _var_name,
              'mean_initial_forcing': 'average\ninitial forcing intensity',
              'initial_forcing_TX': 'initial forcing intensity (TX)',
              'initial_forcing_LA': 'initial forcing intensity (LA)',
              }

    if _z_vis == 'cbar':
        fig, ax = plt.subplots(figsize=figsize)
        cm = plt.cm.get_cmap(_cmap)
        if _num_cbar_bins is not None:
            cm = mpl.colors.LinearSegmentedColormap.from_list('name', cm(np.linspace(0.1, 0.8, _num_cbar_bins)))
            norm = mpl.colors.BoundaryNorm(np.linspace(min(data_xyz[2, :]), max(data_xyz[2, :]), _num_cbar_bins), cm.N)
            sc = ax.scatter(data_xyz[0, :], data_xyz[1, :], c=data_xyz[2, :], norm=norm, cmap=cm, s=4)
            if _plot_regressions:
                for norm_idx in range(len(norm.boundaries) - 1):
                    n0 = norm.boundaries[norm_idx]
                    n1 = norm.boundaries[norm_idx + 1]
                    if norm_idx < len(norm.boundaries) - 1:
                        selection_slice = (data_xyz[2, :] >= n0) & (data_xyz[2, :] < n1)
                    else:
                        selection_slice = (data_xyz[2, :] >= n0) & (data_xyz[2, :] <= n1)
                    x_vals = data_xyz[0, :][selection_slice]
                    y_vals = data_xyz[1, :][selection_slice]
                    reg = stats.linregress(x_vals, y_vals)
                    ax.plot(np.array(x_vals), reg.intercept + reg.slope * np.array(x_vals), #color=cm(norm(n0)),
                            linewidth=0.5, color='k')
                    print("bin=[{}, {}], n={}, r={}, p={}, m={}, b={}".format(n0, n1, len(x_vals), reg.rvalue,
                                                                              reg.pvalue, reg.slope, reg.intercept))
        else:
            sc = ax.scatter(data_xyz[0, :], data_xyz[1, :], c=data_xyz[2, :], vmin=min(data_xyz[2, :]),
                            vmax=max(data_xyz[2, :]), cmap=cm, s=4)
        cb = fig.colorbar(sc)
        cb.set_label(labels[_z_dim])
    elif _z_vis == 'scatter_size':
        fig, ax = plt.subplots(figsize=figsize)
        s_max = 20
        s_min = 2
        z_norm = (data_xyz[2, :] / max(data_xyz[2, :])) * (s_max - s_min) + s_min
        ax.scatter(data_xyz[0, :], data_xyz[1, :], s=z_norm)
    elif _z_vis == '3d':
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        grid_res_x = np.mean(abs(np.unique(data_xyz[0])[1:] - np.unique(data_xyz[0])[:-1]))
        grid_res_y = np.mean(abs(np.unique(data_xyz[1])[1:] - np.unique(data_xyz[1])[:-1]))
        xi = np.arange(min(data_xyz[0, :]), max(data_xyz[0, :]), grid_res_x)
        yi = np.arange(min(data_xyz[1, :]), max(data_xyz[1, :]), grid_res_y)
        xx, yy = np.meshgrid(xi, yi)
        zz = griddata((data_xyz[0, :], data_xyz[1, :]), -data_xyz[2, :], (xi[None, :], yi[:, None]), method='linear')
        # zz = np.full(xx.shape, np.nan)
        # zz = np.full(xx.shape, 0)
        # for idx, _z in enumerate(zi):
        #     _x = xi[0, idx]
        #     _y = yi[1, idx]
        #     zz[np.where(xx[0, :] == _x)[0], np.where(yy[:, 0] == _y)[0]] = _z
        # ax.plot_wireframe(xx, yy, zz)
        vmin = zz[~np.isnan(zz)].min()
        vmax = zz[~np.isnan(zz)].max()
        p = ax.plot_surface(xx, yy, zz, cmap='hot', vmin=vmin, vmax=vmax)

        ax.set_position([0.05, 0.05, 0.95, 1.05])
        ax.tick_params(pad=0, axis='z', labelsize=7)
        ax.tick_params(pad=-5, axis='x', labelsize=7)
        ax.tick_params(pad=-5, axis='y', labelsize=7)
        ax.view_init(20, -140)
        # for angle in range(0, 360):
        #     ax.view_init(30, angle)
        #     plt.draw()
        #     plt.pause(0.001)
        #     print(angle)
        ax.xaxis.labelpad = -3
        ax.yaxis.labelpad = -3
        ax.zaxis.labelpad = -1.5

        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(labels[_z_dim], rotation=90)

        if _hide_every_other_ticklabel:
            for label in ax.xaxis.get_ticklabels()[::2]:
                label.set_visible(False)
            for label in ax.yaxis.get_ticklabels()[::2]:
                label.set_visible(False)

    ax.set_xlabel(labels[_x_dim])
    ax.set_ylabel(labels[_y_dim])

    if _z_vis != '3d':
        plt.tight_layout()

    if not _xlabels:
        ax.set_xticklabels([])
        ax.set_xlabel(None)

    if not _ylabels:
        ax.set_yticklabels([])
        ax.set_ylabel(None)

    if _numbering is not None:
        fig.text(0, min(ax.get_position().y1, 0.98), _numbering, fontweight='bold', ha='left', va='center')
    plt.show()


if __name__ == '__main__':
    # plot_initial_claims(_outfile="../figures/harvey_initial_claims.pdf")
    pass
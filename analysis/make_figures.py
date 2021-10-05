import copy
import json
import pickle
import string

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import os
import sys

import tqdm
from matplotlib import transforms, ticker, patches
from matplotlib import cm

sys.path.append("../")

from scipy import stats
from scipy.interpolate import griddata
from netCDF4 import Dataset

from analysis.map import make_map, create_colormap

from data.calc_initial_forcing_intensity_HARVEY import plot_polygon, load_hwm, alpha_shape, alpha
from analysis.dataformat import AggrData, clean_regions
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

SLOPE_COLORS = plt.cm.get_cmap('Set2')


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
    plt.legend(frameon=False)  # (loc='upper right')
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
    us_states_shp = gpd.read_file(os.path.join(rootdir, 'data/external/gadm36_USA_shp/gadm36_USA_1.shp')).to_crs(
        epsg=3663)
    us_county_shp = gpd.read_file(os.path.join(rootdir, 'data/external/gadm36_USA_shp/gadm36_USA_2.shp')).to_crs(
        epsg=3663)
    us_county_shp[(us_county_shp['NAME_1'].isin(states)) & (
        ~us_county_shp['HASC_2'].isin(np.concatenate(list(affected_counties.values()))))].plot(ax=ax1,
                                                                                               facecolor='lightgrey')
    us_states_shp[us_states_shp['NAME_1'].isin(states)].plot(ax=ax1, edgecolor='k', facecolor='none', linewidth=1)
    for ax_idx, radius_extension in enumerate(radius_extensions):
        plot_polygon(concave_hull.buffer(radius_extension), ax1, _fc='none', _ec=cmap(ax_idx))
        us_county_shp[us_county_shp['HASC_2'].isin(affected_counties[radius_extension])].plot(ax=ax1,
                                                                                              color=cmap(ax_idx))
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
    initial_forcing_intensities = json.load(
        open(os.path.join(rootdir, "data/generated/initial_forcing_params.json"), "rb"))
    affected_counties = json.load(open(os.path.join(rootdir, 'data/generated/affected_counties.json'), 'rb'))
    for key in list(affected_counties.keys()):
        affected_counties[int(key)] = affected_counties.pop(key)
    re = [int(re) for re in affected_counties.keys()]
    for _s in ['LA', 'TX']:
        m_f0_i = initial_forcing_intensities['params'][_s]['m']
        c_f0_i = initial_forcing_intensities['params'][_s]['c']
        print(m_f0_i, c_f0_i)
        ax.plot([re[0], re[-1]], [m_f0_i * re[0] + c_f0_i, m_f0_i * re[-1] + c_f0_i], '--k')
        if _s == 'LA':
            y_pos = 0
            va = 'bottom'
        elif _s == 'TX':
            y_pos = 0.75
            va = 'top'
        ax.text(1, y_pos, "y={0:1.2f}".format(m_f0_i * 1e6) + r"$\cdot 10^{-3}$" + "x+{0:1.2f}".format(c_f0_i), ha='right', va=va,
                transform=ax.transAxes)
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


def prepare_heatmap_figure(_data: AggrData, _type: str, _x_ax: bool, _scale_factor=1.0, gmt_anomaly_0=0,
                           _sst_gmt_factor=0.5, _numbering=None, _y_ax_precision=3):
    # fig_width = MAX_FIG_WIDTH_WIDE * 0.8 * _scale_factor
    fig_width = MAX_FIG_WIDTH_NARROW * _scale_factor
    if _x_ax:
        # fig_height = MAX_FIG_WIDTH_WIDE * _scale_factor * 0.5
        fig_height = MAX_FIG_WIDTH_NARROW * _scale_factor * 0.5 / 0.6
        ax_bbox_y1 = 0.17 / _scale_factor
    else:
        # fig_height = MAX_FIG_WIDTH_WIDE * _scale_factor * 0.45
        fig_height = MAX_FIG_WIDTH_NARROW * _scale_factor * 0.45 / 0.6
        ax_bbox_y1 = 0.05 / _scale_factor
    cbar_bbox_x1 = 0.21 / _scale_factor
    cbar_width = 0.02
    dist_ax_cbar = 0.02
    ax_bbox_x1 = cbar_bbox_x1 + cbar_width + dist_ax_cbar
    ax_bbox_x2 = 1 - 0.12 / _scale_factor
    ax_width = ax_bbox_x2 - ax_bbox_x1
    ax_height = 0.99 - ax_bbox_y1
    ax_bbox = (ax_bbox_x1, ax_bbox_y1, ax_width, ax_height)
    cbar_bbox = (cbar_bbox_x1, ax_bbox_y1, cbar_width, ax_height)
    print(ax_bbox, "\n", cbar_bbox)
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax, cbar_ax = fig.add_axes(ax_bbox), fig.add_axes(cbar_bbox)
    # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.{}f'.format(_y_ax_precision)))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(8))
    if _type == 'heatmap_cut':
        cbar_ax.remove()
        ax.xaxis.set_ticks_position('none')
    if _type == 'heatmap':
        ax1 = ax.twinx()
        ax1.tick_params(axis='y', labelrotation=0)
        ax1.set_ylim(-0.5, len(_data.get_lambda_axis()) - 0.5)
        ax1.set_yticks(np.arange(0, len(_data.get_lambda_axis()), 1))
        ax1.set_yticklabels([int(l) for l in _data.get_lambda_axis() / 1e3])
        ax.set_ylim(-0.5, len(_data.get_lambda_axis()) - 0.5)
        ax.set_yticks([])
        ax.set_yticklabels([])
    if _x_ax:
        ax.set_xlim(-0.5, len(_data.get_duration_axis()) - 0.5)
        ax.set_xlim(-0.5, len(_data.get_duration_axis()) - 0.5)
        ax.set_xticks(np.arange(0, len(_data.get_duration_axis()), 1 / _data.dT_stepwidth * _sst_gmt_factor))
        # ax.set_xticklabels(["{0:1.2f}".format(gmt_anomaly_0 + d / _sst_gmt_factor) for d in _data.get_duration_axis()])
        ax.set_xticklabels(np.arange(0, (len(_data.get_duration_axis()) - 1) * _data.dT_stepwidth / _sst_gmt_factor + 1e-10).astype(int))
        # ax.tick_params(axis='x', labelrotation=90)
        ax.set_xlabel(r'$\Delta GMT$ (difference to $GMT_{2017}$ in °C)')
    else:
        ax.set_xlabel('')
        ax.set_xticklabels([])
        # ax.set_xlim(0, len(_data.get_duration_axis()) - 0.5)
        ax.set_xticks(np.arange(0, len(_data.get_duration_axis()), 1))
    if _numbering is not None:
        fig.text(0, 1, _numbering, ha='left', va='top', fontweight='bold')
    return fig, ax, cbar_ax


def make_heatmap(_data: AggrData, _agg_method: str, base_point=None, contour_distance=None, _gauss_filter=False,
                 _gauss_sigma=1, _gauss_truncate=1, _outfile=None, _slopes=None, _plot_cuts=True, _label=None,
                 _sst_gmt_factor=0.5, _data_division=1.0, _numbering=None, _vmin=None, _vmax=None, _slope_data=None,
                 _inplot_info=True, _y_ax_precision=3, **kwargs):
    if _data.shape[:3] != (1, 1, 1) or _data.shape[-1] != 1 or (
            _slope_data is not None and (_slope_data.shape[:3] != (1, 1, 1) or _slope_data.shape[-1] != 1)):
        raise ValueError("All dimensions of the datasets except lambda and duration.")
    numbering_heatmap = None
    numbering_cut = None
    if _numbering is not None:
        numbering_heatmap, numbering_cut = _numbering[0], _numbering[1]
    fig, ax, cbar_ax = prepare_heatmap_figure(_data, _type='heatmap', _x_ax=(not _plot_cuts),
                                              _sst_gmt_factor=_sst_gmt_factor, _numbering=numbering_heatmap,
                                              _y_ax_precision=_y_ax_precision)
    _data_aggregated = copy.deepcopy(_data.clip(1))
    if _agg_method == 'sum':
        _data_aggregated.data_capsule.data = _data.get_data().sum(axis=-1, keepdims=True)
    elif _agg_method == 'min':
        _data_aggregated.data_capsule.data = _data.get_data().min(axis=-1, keepdims=True)
    elif _agg_method == 'max':
        _data_aggregated.data_capsule.data = _data.get_data().max(axis=-1, keepdims=True)
    data_array = _data.data.reshape((len(_data.get_lambda_axis()), len(_data.get_duration_axis())))
    data_array /= _data_division
    data_filtered = gaussian_filter(data_array, sigma=_gauss_sigma, mode='nearest', truncate=_gauss_truncate)
    if _gauss_filter:
        plot_data = data_filtered
    else:
        plot_data = data_array
    if _slope_data is not None and _vmin is None:
        _vmin = min(_data.data.min(), _slope_data.data.min())
    if _slope_data is not None and _vmax is None:
        _vmax = max(_data.data.max(), _slope_data.data.max())
    norm = mpl.colors.Normalize(vmin=_vmin, vmax=_vmax)
    # cmap = cm.get_cmap('viridis')
    cmap = cm.get_cmap('Oranges_r')
    im = ax.imshow(plot_data, origin='lower', aspect='auto', norm=norm, cmap=cmap)
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar_ax.yaxis.set_ticks_position('left')
    cbar_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.{}f'.format(_y_ax_precision)))
    cbar_ax.yaxis.set_major_locator(ticker.MaxNLocator(8))
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
    if _slopes is None and _slope_data is not None:
        _slopes = [s * _sst_gmt_factor for s in _slope_data.slope_meta.keys()]
    if _slopes is not None:
        _slopes = sorted(_slopes, reverse=False)
        d_i = (_data_aggregated.get_lambda_axis()[-1] - _data_aggregated.get_lambda_axis()[0]) / (len(
            _data_aggregated.get_lambda_axis()) - 1)
        d_d = (_data_aggregated.get_duration_axis()[-1] - _data_aggregated.get_duration_axis()[0]) / (len(
            _data_aggregated.get_duration_axis()) - 1)
        print(d_i, d_d)
        for idx, slope in enumerate(_slopes):
            m = slope * d_d / d_i / _sst_gmt_factor  # slopes are in km/degC GMT change -> translate into km/degC SST change
            b = base_y - m * base_x
            x_max = min(len(_data_aggregated.get_duration_axis()) - 1,
                        ((len(_data_aggregated.get_lambda_axis()) - 1) - b) / m)
            y_max = m * x_max + b
            ax.plot([base_x, x_max], [m * base_x + b, y_max], c='w')
            ax.text(base_x + (x_max - base_x) / 2, m * (base_x + (x_max - base_x) / 2) + b, string.ascii_uppercase[idx],
                    c='w', va='bottom', ha='right')
    # ax.plot(base_x, base_y, marker='x', markersize=8, color='w')
    # ax.text(base_x, base_y, '\n({0:1.3f})'.format(base_z), color='k', fontsize=FSIZE_SMALL, ha='center', va='top')
    if _slope_data is not None:
        binstep_re = abs(_slope_data.get_lambdavals()[1] - _slope_data.get_lambdavals()[0])
        binstep_dT = abs(_slope_data.get_durationvals()[1] - _slope_data.get_durationvals()[0])
        datastep_re = abs(_data.get_lambdavals()[1] - _data.get_lambdavals()[0])
        datastep_dT = abs(_data.get_durationvals()[1] - _data.get_durationvals()[0])
        rectangle_width = binstep_dT / datastep_dT
        rectangle_height = binstep_re / datastep_re
        binsize = len(list(list(_slope_data.slope_meta.values())[0].values())[0])
        binwidth = np.sqrt(binsize) * rectangle_width
        binheight = np.sqrt(binsize) * rectangle_height
        subpoint_coords = []
        point_coords = []
        for slope, points in _slope_data.slope_meta.items():
            for point, subpoints in points.items():
                point_coords.append(point)
                subpoint_coords = subpoint_coords + subpoints
        subpoint_coords = list(set(subpoint_coords))
        for (d, l) in subpoint_coords:
            x = d / datastep_dT
            y = l / datastep_re
            z = _slope_data.get_durationvals(d).get_lambdavals(l).data.flatten()[0]
            rect = patches.Rectangle((x - rectangle_height / 2, y - rectangle_height / 2), rectangle_width,
                                     rectangle_height,
                                     edgecolor=None,
                                     linewidth=0,
                                     facecolor=cmap(norm(z)),
                                     zorder=3,
                                     )
            ax.add_patch(rect)
        for (d, l) in point_coords:
            x = d / datastep_dT
            y = l / datastep_re
            rect = patches.Rectangle((x - binwidth / 2, y - binheight / 2), binwidth, binheight,
                                     edgecolor='w',
                                     linewidth=1,
                                     facecolor='none',
                                     zorder=4,
                                     )
            ax.add_patch(rect)
    trans = transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
    for line_idx, line in enumerate(_label.split('\n')):
        fig.text(0 + 0.04 * line_idx, 0.5, line, rotation=90, ha='left', va='center', transform=trans)
    fig.text(1, 0.5, 'radius change (km)', rotation=90, ha='right', va='center', transform=trans)
    if _inplot_info:
        sector_name = list(_data.get_sectors().keys())[0]
        if sector_name == 'PRIVSECTORS-MINQ':
            sector_name = 'OTHE'
        ax.text(0.02, 0.98, sector_name, transform=ax.transAxes, ha='left', va='top')
    if _plot_cuts:
        make_heatmap_cut(_data_aggregated, _agg_method, _slopes, _gauss_filter=_gauss_filter, _gauss_sigma=_gauss_sigma,
                         _plot_xax=True, _gauss_truncate=_gauss_truncate, _label=_label, _duration_0=duration_0,
                         _sst_gmt_factor=_sst_gmt_factor, _numbering=numbering_cut, _slope_data=_slope_data,
                         _y_ax_precision=_y_ax_precision, **kwargs)
    # plt.tight_layout()
    # if _outfile is not None:
    #     fig.savefig(_outfile, dpi=300)
    plt.show(block=False)
    return fig, ax, cbar_ax, norm


def make_heatmap_cut(_data: AggrData, _slopes=None, _outfile=None, _gauss_filter=True, _gauss_sigma=1,
                     _gauss_truncate=1, _plot_xax=True, _label=None, _sst_gmt_factor=0.5,
                     _duration_0=0, _numbering=None, _slope_data=None, _y_ax_precision=3):
    if _data.shape[:3] != (1, 1, 1) or _data.shape[-1] != 1 or (
            _slope_data is not None and (_slope_data.shape[:3] != (1, 1, 1) or _slope_data.shape[-1] != 1)):
        raise ValueError("All dimensions of the datasets except lambda and duration.")
    if _slopes is None and _slope_data is None:
        raise ValueError("Must pass either slope array or slope data")
    fig, ax, _ = prepare_heatmap_figure(_data, _type='heatmap_cut', _x_ax=_plot_xax, _sst_gmt_factor=_sst_gmt_factor,
                                        _numbering=_numbering, _y_ax_precision=_y_ax_precision)
    data_array = _data.data.reshape((len(_data.get_lambda_axis()), len(_data.get_duration_axis())))
    if _slopes is None and _slope_data is not None:
        _slopes = [s * _sst_gmt_factor for s in _slope_data.slope_meta.keys()]
    if _gauss_filter:
        data_array = gaussian_filter(data_array, sigma=_gauss_sigma, mode='nearest', truncate=_gauss_truncate)
    interp = RectBivariateSpline(_data.get_lambda_axis(), _data.get_duration_axis(), data_array, s=0)
    if _slope_data is None:
        for idx, slope in enumerate(_slopes):
            durations = _data.get_duration_axis()[_data.get_duration_axis() >= _duration_0]
            x_offset = len(_data.get_duration_axis()[_data.get_duration_axis() < _duration_0])
            lambdas = durations * slope / _sst_gmt_factor  # slopes are in km/degC GMT change. Therefore, translate into km/degC SST change:
            lambdas = lambdas[lambdas <= _data.get_lambda_axis()[-1]]
            durations = durations[:len(lambdas)]
            z = [interp(lambdas[i], durations[i])[0, 0] for i in range(len(durations))]
            ax.plot(np.arange(len(durations)) + x_offset, z, label=string.ascii_uppercase[idx], linewidth=0.5)
            # label=string.ascii_uppercase[idx] + ": {0:1.0f} km / $\degree C$".format(_slopes[idx] / 1e3))
    else:
        datastep_re = abs(_data.get_lambdavals()[1] - _data.get_lambdavals()[0])
        datastep_dT = abs(_data.get_durationvals()[1] - _data.get_durationvals()[0])
        num_slopes = len(_slope_data.slope_meta)
        slope_markers = ['s', 'o', '<', '>', 'x', 'D']
        slope_datapoints = get_slope_means_and_errors(_slope_data)
        for (idx, slope), marker in zip(enumerate(sorted(list(slope_datapoints.keys()), reverse=False)), slope_markers[:num_slopes]):
            mean_vals = slope_datapoints[slope]['mean_vals']
            yerrors = slope_datapoints[slope]['yerrors']
            x_vals = slope_datapoints[slope]['x_vals']
            x_vals = [x / datastep_dT - 2. * datastep_dT * (num_slopes / 2 - 0.5 - idx) for x in x_vals]
            ax.errorbar(x_vals, mean_vals, yerr=yerrors, fmt='o', label=string.ascii_uppercase[idx], linewidth=0.5,
                        markersize=3, color='k', alpha=(1 - 0.2 * idx), marker=marker, capsize=1)
    if _label is not None:
        trans = transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
        for line_idx, line in enumerate(_label.split('\n')):
            fig.text(0 + 0.04 * line_idx, 0.5, line, rotation=90, ha='left', va='center', transform=trans)
    ax.legend(frameon=False)
    plt.show()


def get_slope_means_and_errors(_slope_data: AggrData):
    if _slope_data.shape[:3] != (1, 1, 1) or _slope_data.shape[-1] != 1:
        raise ValueError("All dimensions of the datasets except lambda and duration.")
    res = {}
    for slope, points in _slope_data.slope_meta.items():
        res[slope] = {}
        mean_vals = []
        yerrors = [[], []]
        x_vals = []
        for point, subpoints in points.items():
            num_points = len(subpoints)
            val_sum = 0
            subpoint_vals = []
            for (d, l) in subpoints:
                subpoint_val = _slope_data.get_lambdavals(l).get_durationvals(d).data.flatten()[0]
                val_sum += subpoint_val
                subpoint_vals.append(subpoint_val)
            x_vals.append(point[0])
            mean = val_sum / num_points
            mean_vals.append(mean)
            yerrors[0].append(abs(min(subpoint_vals) - mean))
            yerrors[1].append(abs(max(subpoint_vals) - mean))
        res[slope]['mean_vals'] = np.array(mean_vals)
        res[slope]['yerrors'] = np.array(yerrors)
        res[slope]['x_vals'] = np.array(x_vals)
    return res


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


def make_scatter_plot(_data: AggrData, _clip=365, _x_dim='direct_loss', _y_dim='var', _z_dim='re', _z_vis='cbar',
                      _temperature_offset=1, _hide_every_other_ticklabel=False, _num_cbar_bins=None, _cmap='hot',
                      _plot_regressions=True, _x_scale=None, _y_scale=None, _z_scale=None, _numbering=None,
                      _xlabels=True, _ylabels=True):
    if _data.get_sim_duration() != 1:
        raise ValueError('Must pass data with only one timestep')
    # if len(_data.get_vars()) != 1:
    #     raise ValueError('Must pass data with only one variable')
    if len(_data.get_regions()) != 1:
        raise ValueError('Must pass data with only one region')
    if len(_data.get_sectors()) != 1:
        raise ValueError('Must pass data with only one sector')
    # if _x_dim != 'var' and _y_dim != 'var' and _z_dim != 'var':
    #     raise ValueError('One dimension must be the data variable.')
    if _x_dim == _y_dim or _y_dim == _z_dim or _z_dim == _x_dim:
        raise ValueError('all dimensions must be different.')
    figsize = (MAX_FIG_WIDTH_NARROW, 0.9 * MAX_FIG_WIDTH_NARROW)
    _data = _data.clip(_clip)
    data_xyz = [[], [], []]
    for _d_idx, _d in enumerate(_data.get_duration_axis()):
        for _l_idx, _l in enumerate(_data.get_lambda_axis()):
            temp = {
                't_r': _data.scaled_scenarios[(_l, _d)]['params']['all']['t_r'],
                'f_r': _data.scaled_scenarios[(_l, _d)]['params']['all']['f_r'],
                'f_0': _data.scaled_scenarios[(_l, _d)]['params']['all']['f_0'],
                'direct_loss': _data.scaled_scenarios[(_l, _d)]['params']['all']['direct_loss'],
                'tau_TX': _data.scaled_scenarios[(_l, _d)]['params']['US.TX']['tau'],
                'tau_LA': _data.scaled_scenarios[(_l, _d)]['params']['US.LA']['tau'],
                'f_0_TX': _data.scaled_scenarios[(_l, _d)]['params']['US.TX']['f_0'],
                'f_0_LA': _data.scaled_scenarios[(_l, _d)]['params']['US.LA']['f_0'],
                're': int(_l / 1e3),
                'dT': _d + _temperature_offset,
            }
            for dim, (variable, scale) in enumerate(zip([_x_dim, _y_dim, _z_dim], [_x_scale, _y_scale, _z_scale])):
                if variable in temp.keys():
                    data_xyz[dim].append(temp[variable])
                else:
                    data_xyz[dim].append(
                        _data.get_vars(variable).get_lambdavals(_l).get_durationvals(_d).get_data().flatten()[0])
                if scale is not None:
                    data_xyz[dim] = data_xyz[dim] / scale
    data_xyz = np.array(data_xyz)
    labels = {
        're': 'radius extension\n(km)',
        'dT': 'global mean temperature\nanomaly (°C)',
        'f_0': 'average\ninitial forcing intensity',
        'f_0_TX': 'initial forcing intensity (TX)',
        'f_0_LA': 'initial forcing intensity (LA)',
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
                    ax.plot(np.array(x_vals), reg.intercept + reg.slope * np.array(x_vals),  # color=cm(norm(n0)),
                            linewidth=0.5, color='k')
                    print("bin=[{}, {}], n={}, r={}, p={}, m={}, b={}".format(n0, n1, len(x_vals), reg.rvalue,
                                                                              reg.pvalue, reg.slope, reg.intercept))
        else:
            sc = ax.scatter(data_xyz[0, :], data_xyz[1, :], c=data_xyz[2, :], vmin=min(data_xyz[2, :]),
                            vmax=max(data_xyz[2, :]), cmap=cm, s=4)
        cb = fig.colorbar(sc)
        cb.set_label(labels.get(_z_dim, _z_dim))
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
        vmin = zz[~np.isnan(zz)].min()
        vmax = zz[~np.isnan(zz)].max()
        p = ax.plot_surface(xx, yy, zz, cmap='hot', vmin=vmin, vmax=vmax)
        ax.set_position([0.05, 0.05, 0.95, 1.05])
        ax.tick_params(pad=0, axis='z', labelsize=7)
        ax.tick_params(pad=-5, axis='x', labelsize=7)
        ax.tick_params(pad=-5, axis='y', labelsize=7)
        ax.view_init(20, -140)
        ax.xaxis.labelpad = -3
        ax.yaxis.labelpad = -3
        ax.zaxis.labelpad = -1.5
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(labels.get(_z_dim, _z_dim), rotation=90)
        if _hide_every_other_ticklabel:
            for label in ax.xaxis.get_ticklabels()[::2]:
                label.set_visible(False)
            for label in ax.yaxis.get_ticklabels()[::2]:
                label.set_visible(False)
    ax.set_xlabel(labels.get(_x_dim, _x_dim))
    ax.set_ylabel(labels.get(_y_dim, _y_dim))
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


def make_agent_var_global_map(_data, _sector='MINQ', _variable='incoming_demand', _dt=0, _re=0, _exclude_regions=None, _t_0=4,
                              _t_agg=365, _cbar_lims=None, _numbering=None, _outfile=None, _symmetric_cbar=True):
    if _variable not in _data.get_vars():
        raise ValueError("passed dataset does not contain variable {}".format(_variable))
    _data = copy.deepcopy(_data.get_vars(_variable))
    if '-' in _sector:
        sec1, sec2 = _sector.split('-')
        if sec1 not in _data.get_sectors() or sec2 not in _data.get_sectors():
            raise ValueError("Either {} or {} could not be found in sectors.".format(sec1, sec2))
        sec1_idx = np.where(np.array(list(_data.get_sectors().keys())) == sec1)[0][0]
        sec2_idx = np.where(np.array(list(_data.get_sectors().keys())) == sec2)[0][0]
        sec_data = _data.data[:, :, sec1_idx:sec1_idx + 1, ...] - _data.data[:, :, sec2_idx:sec2_idx + 1, ...]
        _data.data_capsule.data = np.concatenate((_data.data, sec_data), axis=2)
        _data.data_capsule.sectors[_sector] = copy.deepcopy(_data.data_capsule.sectors[sec1])
        _data.data_capsule.sectors[_sector].remove(sec2)
    elif _sector not in _data.get_sectors():
        raise ValueError("Sector {} not in data".format(_sector))
    regions = list(set(_data.get_regions().keys()) - set(WORLD_REGIONS.keys()))
    if _exclude_regions is not None:
        regions = list(set(regions) - set(_exclude_regions))
    _data = _data.clip(_t_0, _t_0 + _t_agg)
    _data = _data.get_regions(regions)
    _data = _data.calc_change_to_baseline(mode='absolute', _aggregate=True)
    data_array = _data.get_sectors(_sector).get_lambdavals(_re).get_durationvals(_dt).get_data().flatten()
    print('Total {} loss: {}'.format(_variable, data_array[data_array < 0].sum()))
    print('Total {} gains: {}'.format(_variable, data_array[data_array >= 0].sum()))
    data_array[data_array < 0] = data_array[data_array < 0] / abs(sum(data_array[(data_array < 0) & ~data_array.mask]))
    data_array[data_array >= 0] = data_array[data_array >= 0] / abs(
        sum(data_array[(data_array >= 0) & ~data_array.mask]))
    data_array = data_array * 100
    for r, d in sorted(list(zip(_data.get_regions(), data_array)), key=lambda x: x[1]):
        print(r, '{0:1.3f}'.format(d))
    if _cbar_lims is None:
        _cbar_lims = [data_array.min(), data_array.max()]
    if _symmetric_cbar:
        positive_scale_factor = abs(_cbar_lims[0]) / _cbar_lims[1]
        data_array[data_array >= 0] = data_array[data_array >= 0] * positive_scale_factor
        _cbar_lims[1] *= positive_scale_factor
    cm = create_colormap('custom',
                         ['purple', "white", 'orange'],
                         xs=[0, (abs(_cbar_lims[0])) / (_cbar_lims[1] - _cbar_lims[0]), 1], # alphas=[0.5, 0, 0.5]
                         )
    fig = plt.figure(figsize=(MAX_FIG_WIDTH_WIDE, MAX_FIG_WIDTH_WIDE * 0.44))
    gs = plt.GridSpec(1, 2, width_ratios=[1, 0.03])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    patchespickle_file = "/home/robin/repos/hurricanes_hindcasting_remake/global_map/map_robinson_0.1simplified.pkl.gz"
    make_map(patchespickle_file=patchespickle_file,
             regions=_data.get_regions(),
             data=data_array,
             y_ticks=None,
             y_label=' ',
             numbering=None,
             numbering_fontsize=FSIZE_TINY,
             extend_c="both",
             ax=ax,
             cax=cax,
             cm=cm,
             y_label_fontsize=FSIZE_SMALL,
             y_ticks_fontsize=FSIZE_SMALL,
             ignore_regions=_exclude_regions,
             lims=None,
             only_usa=False,
             v_limits=_cbar_lims,
             show_cbar=True,
             )
    if _symmetric_cbar:
        cax_ticklabels = []
        for tick in cax.get_yticks():
            tick = float(tick)
            if tick > 0:
                tick = tick / positive_scale_factor
            if abs(tick) >= 10:
                tick_formatter = '{:1.0f}'
            else:
                tick_formatter = '{:1.1f}'
            cax_ticklabels.append(tick_formatter.format(abs(tick)))
        cax.set_yticklabels(cax_ticklabels)
    plt.tight_layout()
    trans = transforms.blended_transform_factory(fig.transFigure, cax.transAxes)
    fig.text(0.98, 0.5, 'share (%) of global', ha='right', va='center', rotation=90, transform=trans)
    fig.text(1, 0.5, r'losses   $\longleftrightarrow$   gains', ha='right', va='center', rotation=90, transform=trans)
    if _numbering is not None:
        fig.text(0, min(ax.get_position().y1, 0.98), _numbering, fontweight='bold', ha='left', va='center')
    if _outfile is not None:
        plt.savefig(_outfile, dpi=300)


def generate_flow_dataset(_t0=4, _t_agg=365, _sector='MINQ'):
    data_dir = "/home/robin/repos/harvey_scaling/data/acclimate_output/HARVEY_econYear2015_dT_0_2.5_0.125__re0_100000.0_5000.0__ccFactor1.07/2021-08-11__flow_output/"
    in_data = Dataset(data_dir + "HARVEY_dT0.00_re0_output_with_flows.nc")
    # in_data = Dataset(data_dir + "HARVEY_dT0.00_re0_sent_flow.nc")
    regions = in_data['region'][:]
    sectors = in_data['sector'][:]
    var_arrays = []
    var_names = []
    s_idx = np.where(sectors == _sector)[0][0]
    for var in ['sent_flow', 'demand_request']:
        if var == 'demand_request':
            var_data = in_data['flows/{}'.format(var)][_t0:_t0 + _t_agg, :, :, s_idx, :].transpose(
                (0, 2, 1, 3))  # .sum(axis=1).transpose((1, 2, 0))
            var = 'region_demand_request_to_{}'.format(_sector)
        elif var == 'sent_flow':
            var_data = in_data['flows/{}'.format(var)][_t0:_t0 + _t_agg, s_idx, :, :, :]
        var_data_sum = np.zeros((_t_agg, len(regions), len(regions)))
        for s_idx in range(len(sectors)):
            var_data_sum += np.ma.filled(var_data[..., s_idx, :], 0)
        var_data = var_data_sum.transpose((1, 2, 0)).reshape((1, len(regions), len(regions), 1, 1, 1, _t_agg))
        var_names.append('{}_global'.format(var))
        var_arrays.append(var_data.sum(axis=2))
        domestic_data = np.ma.masked_all((1, len(regions), 1, 1, 1, _t_agg))
        for r_idx in range(len(regions)):
            domestic_data[0, r_idx, ...] = var_data[:, r_idx, r_idx, ...]
        var_names.append('{}_domestic'.format(var))
        var_arrays.append(domestic_data)
    region_dict = {}
    for r in regions:
        region_dict[r] = [r]
    result = AggrData(np.ma.masked_all((0, len(regions), 1, 1, 1, _t_agg)), np.array([]), region_dict,
                      {_sector: [_sector]},
                      np.array([0]), np.array([0]))
    for var_name, var_array in zip(var_names, var_arrays):
        result.add_var(var_array, var_name, _inplace=True)
    return result


def make_flow_var_global_map(_sector='MINQ', _variable='demand_request', _exclude_regions=None, _t_0=4,
                             _t_agg=365, _cbar_lims=None, _numbering=None, _outfile=None):
    data_dir = "/home/robin/repos/harvey_scaling/data/acclimate_output/HARVEY_econYear2015_dT_0_2.5_0.125__re0_100000.0_5000.0__ccFactor1.07/2021-08-11__flow_output/"
    if os.path.exists(data_dir + "{}_anomalies_{}.pk".format(_variable, _sector)):
        data_anomalies = pickle.load(open(data_dir + "{}_anomalies_{}.pk".format(_variable, _sector), 'rb'))
        regions = [i[0] for i in data_anomalies]
        data_anomalies = [i[1] for i in data_anomalies]
    else:
        data_path = data_dir + "HARVEY_dT0.00_re0_{}.nc".format(_variable)
        dataset = Dataset(data_path)
        regions = dataset['region'][:]
        if _exclude_regions is not None:
            regions = list(set(regions) - set(_exclude_regions))
        sector_idx = np.where(dataset['sector'][:] == _sector)[0][0]
        data_anomalies = pd.DataFrame(columns=['total_{}'.format(_variable), 'non_domestic_{}'.format(_variable)],
                                      index=regions)
        for r in tqdm.tqdm(regions):
            r_idx = np.where(dataset['region'][:] == r)[0][0]
            if _variable == 'demand_request':
                r_flows = dataset['flows/{}'.format(_variable)][_t_0:_t_0 + _t_agg, :, r_idx, sector_idx, :].sum(axis=1)
            elif _variable == 'sent_flow':
                r_flows = dataset['flows/{}'.format(_variable)][_t_0:_t_0 + _t_agg, sector_idx, r_idx, :, :].sum(axis=1)
            total_flow = r_flows.sum(axis=1)
            non_domestic_flow = np.delete(r_flows, r_idx, axis=1).sum(axis=1)
            data_anomalies.loc[r] = [total_flow.sum() - (total_flow[0] * len(total_flow)),
                                     non_domestic_flow.sum() - (non_domestic_flow[0] * len(non_domestic_flow))]
        pickle.dump(list(zip(regions, data_anomalies)),
                    open(data_dir + "{}_anomalies_{}.pk".format(_variable, _sector), 'wb'))
    # dataset[dataset < 0] = dataset[dataset < 0] / abs(sum(dataset[dataset < 0]))
    # dataset[dataset >= 0] = dataset[dataset >= 0] / abs(sum(dataset[dataset >= 0]))
    # for r, d in sorted(list(zip(regions, dataset)), key=lambda x: x[1]):
    #     print(r, '{0:1.3f}'.format(d))
    #
    cm = create_colormap(
        'custom',
        ['red', "white", 'blue'],
        xs=[0, (abs(min(data_anomalies))) / (max(data_anomalies) - min(data_anomalies)), 1]
    )
    fig = plt.figure(figsize=(MAX_FIG_WIDTH_WIDE, MAX_FIG_WIDTH_WIDE * 0.44))
    gs = plt.GridSpec(1, 2, width_ratios=[1, 0.03])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    patchespickle_file = "/home/robin/repos/hurricanes_hindcasting_remake/global_map/map_robinson_0.1simplified.pkl.gz"
    # if _variable == 'incoming_demand':
    #     ylabel = 'incoming demand anomaly (relative to total change)'
    # elif _variable == 'production':
    #     ylabel = 'production anomaly (relative to total change)'
    # elif _variable == 'demand':
    #     ylabel = 'demand anomaly (relative to total change)'
    make_map(patchespickle_file=patchespickle_file,
             regions=regions,
             data=data_anomalies,
             y_ticks=None,
             y_label=None,
             numbering=_numbering,
             numbering_fontsize=FSIZE_TINY,
             extend_c="both",
             ax=ax,
             cax=cax,
             cm=cm,
             y_label_fontsize=FSIZE_TINY,
             y_ticks_fontsize=FSIZE_TINY,
             ignore_regions=_exclude_regions,
             lims=None,
             only_usa=False,
             v_limits=(min(data_anomalies), max(data_anomalies)),
             show_cbar=True,
             )
    plt.tight_layout()
    # if _outfile is not None:
    #     plt.savefig(_outfile, dpi=300)


def plot_gains_and_losses(_data: AggrData, _slopes=None, _region_group=None, _gauss_filter=False, _gauss_sigma=1,
                          _gauss_truncate=1, _sst_gmt_factor=0.5, _ylabel_divisor=1.0, _outfile=None, _numbering=None,
                          _x_label=True, _legend=False):
    if _data.get_sim_duration() != 1:
        raise ValueError('Must pass data with only one timestep')
    if len(_data.get_vars()) != 1:
        raise ValueError('Must pass data with only one variable')
    if len(_data.get_sectors()) != 1:
        raise ValueError('Must pass data with only one sector')
    regional_data = {}
    if _region_group is None:
        _region_group = 'WORLD'
    subregions = list(set(WORLD_REGIONS[_region_group]) - set(WORLD_REGIONS.keys()))
    data_array = _data.get_regions(subregions).data / _ylabel_divisor
    data_array = data_array.reshape((len(subregions), len(_data.get_lambda_axis()), len(_data.get_duration_axis())))
    data_gains = copy.deepcopy(data_array)
    data_losses = copy.deepcopy(data_array)
    data_gains[data_gains < 0] = 0
    data_losses[data_losses > 0] = 0
    data_gains = data_gains.sum(axis=0)
    data_losses = data_losses.sum(axis=0)
    if _gauss_filter:
        data_gains = gaussian_filter(data_gains, sigma=_gauss_sigma, mode='nearest', truncate=_gauss_truncate)
        data_losses = gaussian_filter(data_losses, sigma=_gauss_sigma, mode='nearest', truncate=_gauss_truncate)
    interp_gains = RectBivariateSpline(_data.get_lambda_axis(), _data.get_duration_axis(), data_gains, s=0)
    interp_losses = RectBivariateSpline(_data.get_lambda_axis(), _data.get_duration_axis(), data_losses, s=0)
    regional_data[_region_group] = {
        'losses': data_losses,
        'gains': data_gains,
        'interp_losses': interp_losses,
        'interp_gains': interp_gains
    }
    height_scale = 0.85
    if not _x_label:
        height_scale = height_scale * 0.9
    fig, ax = plt.subplots(figsize=(MAX_FIG_WIDTH_NARROW, MAX_FIG_WIDTH_NARROW * height_scale))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(_data.dT_stepwidth / _sst_gmt_factor))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2 = ax.twiny()
    ax2.spines['top'].set_position(('data', 0))
    ax2.tick_params('x', direction='inout', pad=-12)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # ax2.set_yticklabels([abs(t) for t in ax.get_yticks()])
    # ax2.set_xticks(np.arange(0, _data.get_duration_axis().max() / _sst_gmt_factor + 0.1, 0.5))
    ax2.set_xticklabels([])
    dT_sst_list = _data.get_duration_axis()
    for idx, slope in enumerate(_slopes):
        dT_gmt_list = dT_sst_list / _sst_gmt_factor
        re_list = dT_gmt_list * slope  # slopes are in km/degC GMT change. Therefore, translate into km/degC SST change:
        re_list = re_list[re_list <= _data.get_lambda_axis()[-1]]
        global_losses = np.array(
            [regional_data[_region_group]['interp_losses'](re_list[i], dT_sst_list[i])[0, 0] for i in
             range(len(re_list))])
        global_gains = np.array([regional_data[_region_group]['interp_gains'](re_list[i], dT_sst_list[i])[0, 0] for i in
                                 range(len(re_list))])
        dT_gmt_list = dT_gmt_list[:len(re_list)]
        ax.fill_between(dT_gmt_list, global_gains, color='orange', alpha=0.1, linewidth=0)
        ax.fill_between(dT_gmt_list, global_losses, color='purple', alpha=0.1, linewidth=0)
        # ax.plot(dT_gmt_list, global_gains, color='orange', alpha=(1 - 0.2 * idx))
        # ax.plot(dT_gmt_list, global_losses, color='purple', alpha=(1 - 0.2 * idx))
        ax.plot(dT_gmt_list, global_gains + global_losses, color='k', alpha=(1 - 0.5 * idx**0.4),
                label=string.ascii_uppercase[idx])
        print(global_gains / global_losses)
    if _legend:
        ax.legend(loc='upper left', frameon=False)
    # plt.tight_layout()
    ax.set_position((0.25, 0.12 if _x_label else 0.04, 0.7, 0.7 / height_scale))
    trans = transforms.blended_transform_factory(fig.transFigure, ax.transData)
    ax.text(0.06, 0.05, '   gains', rotation=90, transform=trans, va='bottom', ha='left', fontsize=FSIZE_MEDIUM)
    ax.text(0.06, 0.05, 'losses   ', rotation=90, transform=trans, va='top', ha='left', fontsize=FSIZE_MEDIUM)
    if _ylabel_divisor == 1e6:
        unit_label = '(billions of USD)'
    elif _ylabel_divisor == 1e3:
        unit_label = '(millions of USD)'
    elif _ylabel_divisor == 1:
        unit_label = '(thousands of USD)'
    ax.text(0.10, 0.05, unit_label, rotation=90, transform=trans, va='center', fontsize=FSIZE_MEDIUM)
    if _x_label:
        trans = transforms.blended_transform_factory(ax.transAxes, fig.transFigure)
        fig.text(0.5, 0 , r'$\Delta GMT$ (difference to $GMT_{2017}$ in °C)', ha='center', va='bottom', transform=trans)
    else:
        ax.set_xticklabels([])
    if _numbering is not None:
        fig.text(0, 1, _numbering, fontweight='bold', ha='left', va='top')
    if _outfile is not None:
        plt.savefig(_outfile, dpi=300)


def plot_compensation_gap(_data: AggrData, region, _t0=4, _t_agg=365, _sst_gmt_factor=0.5, _legend=True,
                          _numbering=None, _xlabel=True, _ylabel=True, _outfile=None, _inplot_info=True):
    if (_data.shape[0], _data.shape[2]) != (1, 1):
        raise ValueError('Can only pass data with one variable and sector.')
    if _data.slope_meta is None:
        raise ValueError('Must pass slope ensemble dataset.')
    _data = _data.get_regions(WORLD_REGIONS['WORLD']).clip(_t0, _t0 + _t_agg).aggregate('absolute_difference')
    # clean_regions(_data)
    _data_gains = copy.deepcopy(_data)
    _data_losses = copy.deepcopy(_data)
    _data_gains.data_capsule.data[_data.data < 0] = 0
    _data_losses.data_capsule.data[_data.data > 0] = 0
    _data_gains.data_capsule.data = np.abs(_data_gains.data)
    _data_losses.data_capsule.data = np.abs(_data_losses.data)
    clean_regions(_data_gains)
    clean_regions(_data_losses)
    _data = _data_gains
    _data.data_capsule.data = (_data_gains.data / _data_losses.data - 1) * 100
    fig_width = MAX_FIG_WIDTH_NARROW
    fig_height = MAX_FIG_WIDTH_NARROW
    y_scale = 1
    if not _xlabel:
        y_scale = 0.9
    fig, ax = plt.subplots(figsize=(fig_width, fig_height * y_scale))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_width = 0.85
    ax_height = 0.85 / y_scale
    ax_x0 = 0.13
    ax_y0 = 1 - (ax_height + 0.02 / y_scale)
    ax.set_position([ax_x0, ax_y0 , ax_width, ax_height])
    num_slopes = len(_data.slope_meta)
    x_stepwidth = (list(_data.slope_meta[0].keys())[0][0] - list(_data.slope_meta[0].keys())[1][0]) / _sst_gmt_factor
    slope_markers = ['s', 'o', '<', '>', 'x', 'D']
    for (idx, slope), marker in zip(enumerate(sorted(list(_data.slope_meta.keys()), reverse=False)), slope_markers[:num_slopes]):
        datapoints = get_slope_means_and_errors(_data.get_regions(region))
        mean_vals = datapoints[slope]['mean_vals']
        yerrors = datapoints[slope]['yerrors']
        x_vals = datapoints[slope]['x_vals'] / _sst_gmt_factor
        x_vals = [x + x_stepwidth / (2.5 * num_slopes) * (num_slopes / 2 - 0.5 - idx) for x in x_vals]
        ax.errorbar(x_vals, mean_vals, yerr=yerrors, fmt='o', label=string.ascii_uppercase[idx],
                    linewidth=0.5, marker=marker, markersize=3, color='k', alpha=(1 - 0.2 * idx))
    if _legend:
        plt.legend(frameon=False)
    if _inplot_info:
        sector_name = list(_data.get_sectors().keys())[0]
        if sector_name == 'PRIVSECTORS-MINQ':
            sector_name = 'OTHE'
        text = "{}-{}".format(sector_name, region)
        ax.text(0.02, 0.02 / y_scale, text, transform=ax.transAxes, ha='left', va='bottom')
    if _xlabel:
        trans = transforms.blended_transform_factory(ax.transAxes, fig.transFigure)
        fig.text(0.5, 0 / y_scale, r'$\Delta GMT$ (difference to $GMT_{2017}$ in °C)', ha='center', va='bottom', transform=trans)
    if _ylabel:
        trans = transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
        fig.text(0, 0.5 / y_scale, 'compensation gap (% losses)', rotation=90, ha='left', va='center',
                 transform=trans)
    if not _xlabel:
        ax.set_xticklabels([])
    if _numbering is not None:
        trans = transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
        fig.text(0, 1, _numbering, fontweight='bold', ha='left', va='center', transform=trans)
    if _outfile is not None:
        plt.savefig(_outfile, dpi=300)


def plot_global_gain_shares(_data: AggrData, _region_groups, _slopes=None, _gauss_filter=None, _gauss_sigma=1,
                            _gauss_truncate=1, _sst_gmt_factor=0.5, _outfile=None, _numbering=None, _relative=True):
    if _data.get_sim_duration() != 1:
        raise ValueError('Must pass data with only one timestep')
    if len(_data.get_vars()) != 1:
        raise ValueError('Must pass data with only one variable')
    if len(_data.get_sectors()) != 1:
        raise ValueError('Must pass data with only one sector')
    if 'WORLD' not in _region_groups:
        _region_groups.append('WORLD')
    regional_data = {}
    for region_name in _region_groups:
        subregions = list(set(WORLD_REGIONS[region_name]) - {region_name})
        if region_name == 'WORLD':
            subregions = list(set(subregions) - set(WORLD_REGIONS.keys()))
        data_gains = _data.get_regions(subregions).data
        # data_gains = data_gains.reshape((len(subregions), len(_data.get_lambda_axis()), len(_data.get_duration_axis())))
        data_gains = data_gains[0, :, 0, :, :, 0]
        data_gains[data_gains < 0] = 0
        data_gains = data_gains.sum(axis=0)
        if _gauss_filter is not None:
            data_gains = gaussian_filter(data_gains, sigma=_gauss_sigma, mode='nearest', truncate=_gauss_truncate)
        interp_gains = RectBivariateSpline(_data.get_lambda_axis(), _data.get_duration_axis(), data_gains, s=0)
        regional_data[region_name] = {
            'gains': data_gains,
            'interp_gains': interp_gains
        }
    fig, axs = plt.subplots(int(np.sqrt(len(_region_groups))), int(np.ceil(np.sqrt(len(_region_groups)))))
    dT_sst_list = _data.get_duration_axis()
    for region_group, ax in zip(_region_groups, axs.flatten()):
        ax.set_title(region_group)
        for idx, slope in enumerate(_slopes):
            dT_gmt_list = dT_sst_list / _sst_gmt_factor
            re_list = dT_gmt_list * slope  # slopes are in km/degC GMT change. Therefore, translate into km/degC SST change:
            re_list = re_list[re_list <= _data.get_lambda_axis()[-1]]
            global_gains = np.array(
                [regional_data['WORLD']['interp_gains'](re_list[i], dT_sst_list[i])[0, 0] for i in range(len(re_list))])
            region_gains = np.array(
                [regional_data[region_group]['interp_gains'](re_list[i], dT_sst_list[i])[0, 0] for i in
                 range(len(re_list))])
            dT_gmt_list = dT_gmt_list[:len(re_list)]
            if _relative:
                ax.plot(dT_gmt_list, region_gains / global_gains, color='k', alpha=(1 - 0.2 * idx))
            else:
                ax.plot(dT_gmt_list, region_gains, color='k', alpha=(1 - 0.2 * idx))
    plt.tight_layout()


def plot_global_shares_from_slope_dataset(_data: AggrData, _region_groups, _sst_gmt_factor=0.5, _outfile=None,
                                          _numbering=None, _relative=True, _plot_gains=True, _t0=4, _t_agg=365,
                                          _ylim=None, _ylabel=True, _xlabel=True, _bar_reg_label=True,
                                          _inplot_legend=True, _inplot_sector_label=True):
    if (_data.shape[0], _data.shape[2]) != (1, 1):
        raise ValueError('Can only pass data with one variable and sector.')
    _data = _data.get_regions(WORLD_REGIONS['WORLD']).clip(_t0, _t0 + _t_agg).aggregate('absolute_difference')
    if _plot_gains:
        _data.data_capsule.data[_data.data < 0] = 0
    else:
        _data.data_capsule.data[_data.data > 0] = 0
    clean_regions(_data)
    if _relative:
        _data.data_capsule.data = _data.data / _data.get_regions('WORLD').data * 100
    else:
        _data.data_capsule.data = _data.data / 1e6
    y_scale = 1.0
    if not _xlabel:
        y_scale = 0.9
    fig, ax = plt.subplots(figsize=(MAX_FIG_WIDTH_NARROW, MAX_FIG_WIDTH_NARROW * y_scale))
    ax_width = 0.85
    ax_height = 0.85 / y_scale
    ax_x0 = 0.13
    ax_y0 = 1 - (ax_height + 0.02 / y_scale)
    ax.set_position([ax_x0, ax_y0, ax_width, ax_height])
    num_slopes = len(_data.slope_meta.keys())
    bar_width = 0.2
    for slope_idx, slope in enumerate(sorted(list(_data.slope_meta.keys()), reverse=False)):
        pre_mean_vals = None
        for region_idx, region_group in enumerate(_region_groups):
            slope_datapoints = get_slope_means_and_errors(_data.get_regions(region_group))
            mean_vals = slope_datapoints[slope]['mean_vals']
            yerrors = slope_datapoints[slope]['yerrors']
            x_vals = slope_datapoints[slope]['x_vals'] / _sst_gmt_factor
            if slope_idx == 0:
                if _bar_reg_label:
                    xmin = x_vals[0] - (num_slopes / 2 + 1.5) * bar_width
                else:
                    xmin = x_vals[0] - num_slopes / 2 * bar_width
                ax.plot((xmin, x_vals[-1] + num_slopes / 2 * bar_width),
                        (mean_vals[0], mean_vals[0]), c='k', linewidth=0.5)
                text_ypos = pre_mean_vals[0] + (mean_vals[0] - pre_mean_vals[0]) / 2 if pre_mean_vals is not None else mean_vals[0] / 2
                if _bar_reg_label:
                    ax.text(xmin, text_ypos, region_group, ha='left', va='center', rotation=90)
            if pre_mean_vals is not None:
                mean_vals = mean_vals - pre_mean_vals
            x_vals = [x - bar_width * (num_slopes / 2 - 0.5 - slope_idx) for x in x_vals]
            # ax.errorbar(x_vals, mean_vals, yerr=yerrors, fmt='o', label=string.ascii_uppercase[idx], linewidth=1,
            #             markersize=2)
            ax.bar(x_vals, mean_vals, width=bar_width, yerr=yerrors, label=slope, align='center', bottom=pre_mean_vals,
                   color=SLOPE_COLORS(slope_idx), alpha=(1 - region_idx * 0.2))
            pre_mean_vals = pre_mean_vals + mean_vals if pre_mean_vals is not None else mean_vals
            # for (x, y) in zip(x_vals, pre_mean_vals):
            #     ax.plot((x - bar_width / 2, x + bar_width / 2), (y, y), c='k')
            if _inplot_legend:
                rectangle_height = 0.05
                rectangle_width = 0.05
                x = 0.75 + rectangle_width * slope_idx
                y = 0.75 + rectangle_height * region_idx
                rect = patches.Rectangle((x, y), rectangle_width, rectangle_height, edgecolor=None, linewidth=0,
                                         facecolor=SLOPE_COLORS(slope_idx), alpha=(1 - slope_idx * 0.3),
                                         transform=ax.transAxes)
                ax.add_patch(rect)
                if slope_idx == 0:
                    ax.text(x - 0.01, y + rectangle_height / 2, region_group, ha='right', va='center', transform=ax.transAxes)
                if region_idx == 0:
                    ax.text(x + rectangle_width / 2, y - 0.01, string.ascii_uppercase[slope_idx], ha='center', va='top',
                            transform=ax.transAxes)
    if _inplot_sector_label:
        sector_name = list(_data.get_sectors().keys())[0]
        if sector_name == 'PRIVSECTORS-MINQ':
            sector_name = 'OTHE'
        ax.text(0.02, 0.98, sector_name, transform=ax.transAxes, ha='left', va='top')
    if _ylabel:
        if _relative:
            ax.set_ylabel('share of global gains (%)')
        else:
            ax.set_ylabel('absolute gains (bn USD)')
    if _xlabel:
        ax.set_xlabel(r'$\Delta GMT$ (difference to $GMT_{2017}$ in °C)')
    else:
        ax.set_xticklabels([])
    if _ylim is not None:
        ax.set_ylim(_ylim)
    if _numbering is not None:
        fig.text(0, 1, _numbering, ha='left', va='top', fontweight='bold')
    if _outfile is not None:
        plt.savefig(_outfile, dpi=300)


def plot_sector_gain_shares(_data: AggrData, _regions, _sector, _slopes=None, _gauss_filter=None, _gauss_sigma=1,
                            _gauss_truncate=1, _sst_gmt_factor=0.5, _outfile=None, _numbering=None):
    if _data.get_sim_duration() != 1:
        raise ValueError('Must pass data with only one timestep')
    if _sector not in _data.get_sectors() or 'PRIVSECTORS' not in _data.get_sectors():
        raise ValueError('{} or PRIVSECTORS not found in sectors.'.format(_sector))
    if len(_data.get_vars()) != 1:
        raise ValueError('Must pass data with only one variable')
    regional_data = {}
    for region_name in _regions:
        sector_gain_shares = _data.get_regions(region_name).get_sectors(_sector).data / _data.get_regions(
            region_name).get_sectors('PRIVSECTORS').data
        if _gauss_filter is not None:
            sector_gain_shares = gaussian_filter(sector_gain_shares, sigma=_gauss_sigma, mode='nearest',
                                                 truncate=_gauss_truncate)
        sector_gain_shares = sector_gain_shares.reshape((len(_data.get_lambda_axis()), len(_data.get_duration_axis())))
        interp_gain_shares = RectBivariateSpline(_data.get_lambda_axis(), _data.get_duration_axis(), sector_gain_shares,
                                                 s=0)
        regional_data[region_name] = {
            'gain_shares': sector_gain_shares,
            'interp_gain_shares': interp_gain_shares
        }
    fig, axs = plt.subplots(int(np.ceil(np.sqrt(len(_regions)))), int(np.ceil(np.sqrt(len(_regions)))))
    dT_sst_list = _data.get_duration_axis()
    for region_group, ax in zip(_regions, axs.flatten()):
        ax.set_title(region_group)
        for idx, slope in enumerate(_slopes):
            dT_gmt_list = dT_sst_list / _sst_gmt_factor
            re_list = dT_gmt_list * slope  # slopes are in km/degC GMT change. Therefore, translate into km/degC SST change:
            re_list = re_list[re_list <= _data.get_lambda_axis()[-1]]
            sec_gain_shares = np.array(
                [regional_data[region_group]['interp_gain_shares'](re_list[i], dT_sst_list[i])[0, 0] for i in
                 range(len(re_list))])
            dT_gmt_list = dT_gmt_list[:len(re_list)]
            ax.plot(dT_gmt_list, sec_gain_shares, color='k', alpha=(1 - 0.2 * idx))
    plt.tight_layout()


def calc_stats(_data: AggrData, _sector='MINQ', _variable='production', _exclude_regions=None, _t_0=4,
               _t_agg=365, _cbar_lims=None, _numbering=None, _outfile=None, _dt=0, _re=0):
    regions = list(set(_data.get_regions().keys()) - set(WORLD_REGIONS.keys()))
    if _exclude_regions is not None:
        regions = list(set(regions) - set(_exclude_regions))
    _data = _data.clip(_t_0, _t_0 + _t_agg).get_vars(_variable).calc_change_to_baseline(
        mode='absolute', _aggregate=True).get_sectors(_sector).get_lambdavals(_re).get_durationvals(_dt)
    world_data_array = _data.get_regions(regions).data.flatten()
    total_loss = world_data_array[world_data_array < 0].sum()
    total_gains = world_data_array[world_data_array >= 0].sum()
    print('Total {} loss: {}'.format(_variable, total_loss))
    print('Total {} gains: {}'.format(_variable, total_gains))
    stats = pd.DataFrame(columns=['{}_difference'.format(_variable), 'gains', 'losses', 'gains_share', 'loss_share'])
    for r in list(set(WORLD_REGIONS['WORLD'] + list(WORLD_REGIONS.keys()))):
        if r in WORLD_REGIONS.keys():
            region_data = _data.get_regions(list(set(WORLD_REGIONS[r]) - set(WORLD_REGIONS.keys()))).data
        else:
            region_data = _data.get_regions(r).data
        region_gains = region_data[region_data > 0].sum()
        region_loss = region_data[region_data < 0].sum()
        stats.loc[r] = [region_gains + region_loss, region_gains, region_loss, region_gains / total_gains,
                        region_loss / total_loss]
    return stats


def calc_sector_export(_sector='MINQ',
                       _baseline_path="/mnt/cluster_p/projects/acclimate/data/eora/Eora26-v199.82-2015-CHN-USA_naics_disagg.nc"):
    baseline_data = Dataset(_baseline_path)
    flows = baseline_data['flows'][:]
    exports = pd.DataFrame(columns=['export'])
    sector_idx = np.where(baseline_data['sector'][:] == _sector)[0][0]
    for r in tqdm.tqdm(baseline_data['region'][:]):
        region_idx = np.where(baseline_data['region'][:] == r)[0][0]
        rs_from = \
            np.where((baseline_data['index_region'][:] == region_idx) & (baseline_data['index_sector'] == sector_idx))[
                0][0]
        rs_to = np.where((baseline_data['index_region'][:] != region_idx))[0]
        exports.loc[r] = np.ma.filled(flows[rs_from, rs_to], 0).sum()
    for r in ['USA', 'CHN']:
        region_idx = np.where(np.isin(baseline_data['region'][:], WORLD_REGIONS[r]))[0]
        rs_from = \
            np.where(
                np.isin(baseline_data['index_region'][:], region_idx) & (baseline_data['index_sector'] == sector_idx))[
                0]
        rs_to = np.where(~np.isin(baseline_data['index_region'][:], region_idx))[0]
        exports.loc[r] = np.ma.filled(flows[rs_from, :][:, rs_to], 0).sum()
    exports = exports.loc[[i for i in exports.index if i[:3] not in ['CN.', 'US.']]]
    exports['share'] = exports['export'] / exports['export'].sum()
    exports = exports.sort_values(by='export', ascending=False)
    exports['cum_share'] = 0
    for i in range(len(exports)):
        exports.loc[exports.index[i], 'cum_share'] = exports.iloc[:i + 1]['share'].sum()
    return exports


def load_data():
    slope_ensemble_path = "../data/acclimate_output/HARVEY_econYear2015_dT_0.0_2.5_0.5__slopes_0+10000+20000+40000__ccFactor1.07/2021-09-08_18:30:03/"
    full_data_ensemble_path = "../data/acclimate_output/HARVEY_econYear2015_dT_0_2.5_0.125__re0_100000.0_5000.0__ccFactor1.07/2021-08-19_16:30:34__disagg_new/"
    datasets = []
    for ensemble_path in [full_data_ensemble_path, slope_ensemble_path]:
        datacap = pickle.load(open(ensemble_path + "HARVEY_production_t400_MINQ+PRIVSECTORS__data_cap.pk", 'rb'))
        meta = pickle.load(open(ensemble_path + "ensemble_meta.pk", 'rb'))
        if 'slopes' in meta.keys():
            slope_meta = meta['slopes']
        else:
            slope_meta = None
        data = AggrData(datacap, _scaled_scenarios=meta['scaled_scenarios'], _slope_meta=slope_meta)
        data.calc_sector_diff('PRIVSECTORS', 'MINQ', _inplace=True)
        clean_regions(data)
        datasets.append(data)
    return datasets[0], datasets[1]


if __name__ == '__main__':
    # data, slope_data = load_data()
    # make_agent_var_global_map(data, _sector='PRIVSECTORS-MINQ', _variable='production', _numbering='a', _outfile="../figures/maps/global_map_production_change_PRIVSECTORS-MINQ_dt0_re0.pdf", _cbar_lims=[-100, 10], _symmetric_cbar=True)
    # make_agent_var_global_map(data, _sector='MINQ', _variable='production', _numbering='b', _outfile="../figures/maps/global_map_production_change_MINQ_dt0_re0.pdf", _cbar_lims=[-100, 10], _symmetric_cbar=True)
    # plot_initial_claims(_outfile="../figures/harvey_initial_claims.pdf")
    # plot_radius_extension_impact(_outfile="../figures/radius_extension_impact.pdf")
    # make_heatmap(data.get_regions('WORLD_wo_TX_LA').get_vars('production').get_sectors('PRIVSECTORS-MINQ').clip(4, 365 + 4).aggregate('relative_difference'), 'sum', _label='relative production\ndifference( % baseline)', _numbering=['a', 'c'], _slope_data=slope_data.get_regions('WORLD_wo_TX_LA').get_vars('production').get_sectors('PRIVSECTORS-MINQ').clip(4, 365+4).aggregate('relative_difference'), _gauss_filter=False, _sst_gmt_factor=0.5, _plot_cuts=True, _y_ax_precision=3)
    # make_heatmap(data.get_regions('WORLD_wo_TX_LA').get_vars('production').get_sectors('MINQ').clip(4, 365 + 4).aggregate('relative_difference'), 'sum', _label='relative production\ndifference (% baseline)', _numbering=['b', 'd'], _slope_data=slope_data.get_regions('WORLD_wo_TX_LA').get_vars('production').get_sectors('MINQ').clip(4, 365+4).aggregate('relative_difference'), _gauss_filter=False, _sst_gmt_factor=0.5, _plot_cuts=True, _y_ax_precision=2)
    # plot_global_shares_from_slope_dataset(slope_data.get_sectors('PRIVSECTORS-MINQ'), _region_groups=['MINQ25', 'MINQ50', 'MINQ75', 'MINQ95'], _sst_gmt_factor=0.5, _bar_reg_label=False, _numbering='a', _relative = True, _xlabel = False, _ylim = (0, 95), _outfile="../figures/gain_shares/global_gain_shares_PRIVSECTORS-MINQ_rel.pdf")
    # plot_global_shares_from_slope_dataset(slope_data.get_sectors('MINQ'), _region_groups=['MINQ25', 'MINQ50', 'MINQ75', 'MINQ95'], _sst_gmt_factor=0.5, _ylim=(0, 95), _bar_reg_label=False, _inplot_legend=False, _numbering = 'b', _ylabel = False, _xlabel = False, _outfile="../figures/gain_shares/global_gain_shares_MINQ_rel.pdf")
    # plot_global_shares_from_slope_dataset(slope_data.get_sectors('PRIVSECTORS-MINQ'), _region_groups=['MINQ25', 'MINQ50', 'MINQ75', 'MINQ95'], _sst_gmt_factor=0.5, _bar_reg_label=False, _numbering='c', _relative = False, _ylabel = True, _inplot_legend = False, _outfile="../figures/gain_shares/global_gain_shares_PRIVSECTORS-MINQ_abs.pdf")
    # plot_global_shares_from_slope_dataset(slope_data.get_sectors('MINQ'), _region_groups=['MINQ25', 'MINQ50', 'MINQ75', 'MINQ95'], _sst_gmt_factor=0.5, _bar_reg_label=False, _numbering='d', _relative=False, _ylabel = False, _inplot_legend = False, _outfile="../figures/gain_shares/global_gain_shares_MINQ_abs.pdf")
    # plot_gains_and_losses(data.aggregate('absolute_difference').get_sectors('PRIVSECTORS-MINQ'), [0, 5e3, 10e3, 20e3], _region_group='WORLD', _ylabel_divisor=1e6, _numbering='a', _legend=True, _x_label=False, _outfile="../figures/region_gains_and_losses/gains_losses_WORLD_PRIVSECTORS-MINQ.pdf")
    # plot_gains_and_losses(data.aggregate('absolute_difference').get_sectors('MINQ'), [0, 5e3, 10e3, 20e3], _region_group='WORLD', _ylabel_divisor=1e6, _numbering='b', _x_label=False, _outfile="../figures/region_gains_and_losses/gains_losses_WORLD_MINQ.pdf")
    # plot_gains_and_losses(data.aggregate('absolute_difference').get_sectors('PRIVSECTORS-MINQ'), [0, 5e3, 10e3, 20e3], _region_group='USA', _ylabel_divisor=1e6, _numbering='c', _outfile="../figures/region_gains_and_losses/gains_losses_USA_PRIVSECTORS-MINQ.pdf")
    # plot_gains_and_losses(data.aggregate('absolute_difference').get_sectors('MINQ'), [0, 5e3, 10e3, 20e3], _region_group='USA', _ylabel_divisor=1e6, _numbering='d', _outfile="../figures/region_gains_and_losses/gains_losses_USA_MINQ.pdf")
    # plot_compensation_gap(slope_data.get_sectors('PRIVSECTORS-MINQ'), region='WORLD', _numbering='a', _xlabel=False, _outfile="../figures/compensation_gap/compensation_gap_PRIVSECTORS-MINQ_WORLD.pdf")
    # plot_compensation_gap(slope_data.get_sectors('MINQ'), region='WORLD', _numbering='b', _xlabel=False, _legend=False, _ylabel=False, _outfile="../figures/compensation_gap/compensation_gap_MINQ_WORLD.pdf")
    # plot_compensation_gap(slope_data.get_sectors('PRIVSECTORS-MINQ'), region='USA', _numbering='c', _xlabel=True, _legend=False, _outfile="../figures/compensation_gap/compensation_gap_PRIVSECTORS-MINQ_USA.pdf")
    # plot_compensation_gap(slope_data.get_sectors('MINQ'), region='USA', _numbering='d', _xlabel=True, _ylabel=False, _legend=False, _outfile="../figures/compensation_gap/compensation_gap_MINQ_USA.pdf")
    pass


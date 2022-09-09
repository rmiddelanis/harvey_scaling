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
sys.path.append("/home/robin/repos/post-processing/")
from acclimate.dataset import AcclimateOutput
from matplotlib import transforms, ticker, patches
from matplotlib import cm

sys.path.append("/home/robin/repos/harvey_scaling/")

from scipy import stats
from scipy.interpolate import griddata
from netCDF4 import Dataset
import xarray as xr

from scripts.map import make_map, create_colormap

from scripts.calc_initial_forcing_intensity_HARVEY import plot_polygon, load_hwm, alpha_shape, alpha
from scripts.dataformat import AggrData, clean_regions
from scripts.utils import WORLD_REGIONS, cn_gadm_to_iso_code, SECTOR_GROUPS
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.ndimage import gaussian_filter, zoom
import matplotlib as mpl

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

REGION_COLORS = plt.cm.get_cmap('Set2')


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


def plot_radius_extension_map(_numbering=None, _outfile=None, _shape_outpath=None, re_selection=None):
    states = ['Louisiana', 'Texas']
    affected_counties = json.load(open(os.path.join(rootdir, 'data/generated/affected_counties.json'), 'rb'))
    if re_selection is not None:
        for re_drop in [re_ for re_ in list(affected_counties.keys()) if int(re_) not in re_selection]:
            affected_counties.pop(re_drop)
    for key in list(affected_counties.keys()):
        affected_counties[int(key)] = affected_counties.pop(key)
    radius_extensions = [int(re) for re in affected_counties.keys()]
    fig_width = MAX_FIG_WIDTH_NARROW
    fig_height = fig_width * 0.75
    fig, (ax1, cbar_ax) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    cbar_width = 0.02
    cbar_x2 = 1 - (0.15 * MAX_FIG_WIDTH_NARROW / fig_width)
    cbar_x1 = cbar_x2 - cbar_width
    cbar_y1 = 0
    cbar_y2 = 1
    cbar_dist = 0.02
    cbar_ax.set_position([cbar_x1, cbar_y1, cbar_x2 - cbar_x1, cbar_y2 - cbar_y1])
    ax1_bbox_x1 = 0.03
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
    cb2.set_label('Radius change [km]')
    cb2.set_ticks(np.arange(0.5, len(radius_extensions), 1))
    cb2.set_ticklabels([str(int(r / 1e3)) for r in radius_extensions])
    hwm_gdf = load_hwm()
    concave_hull, _ = alpha_shape(hwm_gdf.geometry, alpha=alpha)
    track_shp = gpd.read_file("/home/robin/data/TC_tracks/IBTrACS.since1980.list.v04r00.lines/IBTrACS.since1980.list.v04r00.lines.shp").to_crs(epsg=3663)
    track_shp = track_shp[(track_shp.NAME == 'HARVEY') & (track_shp.SEASON == 2017)]
    us_states_shp = gpd.read_file(os.path.join(rootdir, 'data/external/gadm36_USA_shp/gadm36_USA_1.shp')).to_crs(
        epsg=3663)
    us_states_shp = us_states_shp[us_states_shp['NAME_1'].isin(states)]
    us_county_shp = gpd.read_file(os.path.join(rootdir, 'data/external/gadm36_USA_shp/gadm36_USA_2.shp')).to_crs(
        epsg=3663)
    us_county_shp = us_county_shp[us_county_shp['NAME_1'].isin(states)]
    us_county_shp[(~us_county_shp['HASC_2'].isin(list(affected_counties.values())))].plot(ax=ax1, facecolor='lightgrey')
    us_states_shp.plot(ax=ax1, edgecolor='k', facecolor='none', linewidth=1)
    re_shapes = gpd.GeoDataFrame(radius_extensions, columns=['radius_extension'],
                                 geometry=[concave_hull.buffer(re) for re in radius_extensions])
    re_shapes.set_index('radius_extension', inplace=True)
    re_shapes.set_crs(epsg=3663)
    for ax_idx, re in enumerate(radius_extensions):
        plot_polygon(re_shapes.loc[re].geometry, ax1, _fc='none', _ec=cmap(ax_idx))
        us_county_shp[us_county_shp['HASC_2'].isin(affected_counties[re])].plot(ax=ax1, color=cmap(ax_idx))
    ax1.set_xticks([])
    ax1.set_yticks([])
    hwm_gdf.plot(ax=ax1, markersize=0.4, color='midnightblue', marker='o')
    track_shp.plot(ax=ax1, color='k', linestyle='dotted')
    ylim = (2.55e6, 3.55e6)
    xlim = (7e5, 1.85e6)
    metric = ylim[1] - ylim[0]
    x0, y0 = xlim[0], ylim[0]
    y_pos = 0.2 * metric
    y_radius = 0.03 * metric
    y_distance = 0.05 * metric
    x_pos = 0.6 * metric
    x_radius = 0.06 * metric
    x_distance = 0.01 * metric
    ax1.plot([x0 + x_pos - x_radius / 2, x0 + x_pos + x_radius / 2], [y0 + y_pos, y0 + y_pos], color='k', linestyle='dotted')
    ax1.text(x0 + (x_pos + x_radius + x_distance), y0 + y_pos, 'Harvey track', va='center', ha='left')
    y_pos = y_pos - y_distance
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
    if _shape_outpath is not None:
        hwm_gdf[['latitude', 'longitude', 'geometry']].to_file(_shape_outpath + "/high_water_marks_shapes.shp")
        track_shp.to_file(_shape_outpath + "/storm_track.shp")
        us_states_shp[['HASC_1', 'geometry']].to_file(_shape_outpath + "/state_shapes.shp")
        us_county_shp[['HASC_2', 'geometry']].to_file(_shape_outpath + "/county_shapes.shp")
        re_shapes.to_file(_shape_outpath + "/radius_extension_shapes.shp")
        json.dump(affected_counties, open(_shape_outpath + "affected_counties.json", 'w'))
    plt.show()


def plot_radius_extension_impact(_outfile=None, _numbering=None, store_data=None):
    fig, ax = plt.subplots(figsize=(MAX_FIG_WIDTH_NARROW, 0.75 * MAX_FIG_WIDTH_NARROW))
    initial_forcing_intensities = json.load(
        open(os.path.join(rootdir, "data/generated/initial_forcing_params.json"), "rb"))
    affected_counties = json.load(open(os.path.join(rootdir, 'data/generated/affected_counties.json'), 'rb'))
    for key in list(affected_counties.keys()):
        affected_counties[int(key)] = affected_counties.pop(key)
    re = [int(re) for re in affected_counties.keys()]
    data_output = pd.DataFrame(columns=['TX', 'LA'], index=re)
    data_output.index.name = 'delta_r'
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
        f_vals = [initial_forcing_intensities['points'][str(re)][_s] for re in re]
        ax.scatter(re, f_vals, label=_s, s=5)
        data_output[_s] = f_vals
    ax.set_yticks([0.1 * i for i in range(6)])
    # ax1.set_yticklabels(np.arange(0, 0.55, 0.1))
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, borderpad=0., handletextpad=0.5, handlelength=1)
    ax.set_xlabel('Radius change [km]')
    plt.tight_layout()
    ax.set_position([0.18, ax.get_position().y0, ax.get_position().width - (0.18 - ax.get_position().x0),
                     ax.get_position().height])
    fig.text(-0.22, 0.5, r'$f^{(0)}_s$', fontsize=FSIZE_MEDIUM, ha='left', va='center', transform=ax.transAxes)
    ax.set_xticks(re)
    ax.set_xticklabels(["{}".format(int(_re / 1e3)) for _re in re], rotation=90)
    if _numbering is not None:
        fig.text(0, 1, _numbering, ha='left', va='top', fontweight='bold')
    plt.show()
    if isinstance(_outfile, str):
        fig.savefig(_outfile, dpi=300)
    if store_data is not None:
        data_output.to_csv(store_data)


def prepare_heatmap_figure(_data: AggrData, _type: str, _x_ax: bool, gmt_anomaly_0=0, _xlabel=False,
                           _sst_gmt_factor=0.5, _numbering=None, _y_ax_precision=3):
    fig_width = MAX_FIG_WIDTH_NARROW
    y_scale = 1
    if _type == 'heatmap':
        y_scale = y_scale + 0.05
    if not _xlabel:
        y_scale = y_scale - 0.05
    fig_height = MAX_FIG_WIDTH_NARROW * y_scale * 0.8
    ax_bbox_y1 = (0.15 if _xlabel else 0.1) / y_scale
    cbar_bbox_x1 = 0.21
    cbar_width = 0.02
    dist_ax_cbar = 0.02
    ax_bbox_x1 = cbar_bbox_x1 + cbar_width + dist_ax_cbar
    ax_bbox_x2 = 1 - 0.12
    ax_width = ax_bbox_x2 - ax_bbox_x1
    ax_height = ax_width * fig_width / fig_height
    ax_bbox = (ax_bbox_x1, ax_bbox_y1, ax_width, ax_height)
    cbar_bbox = (cbar_bbox_x1, ax_bbox_y1, cbar_width, ax_height)
    print(ax_bbox, "\n", cbar_bbox)
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax, cbar_ax = fig.add_axes(ax_bbox), fig.add_axes(cbar_bbox)
    # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0f'))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(8))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.{}f'.format(_y_ax_precision)))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1 / _data.dT_stepwidth * _sst_gmt_factor))
    if _type == 'heatmap_cut':
        cbar_ax.remove()
        ax.xaxis.set_ticks_position('none')
        ax.tick_params('x', pad=1)
    if _type == 'heatmap':
        ax1 = ax.twinx()
        ax1.tick_params(axis='y', labelrotation=0)
        ax1.set_ylim(-0.5, len(_data.get_re_axis()) - 0.5)
        ax1.set_yticks(np.arange(0, len(_data.get_re_axis()), 1))
        ax1.set_yticklabels(["{}".format(int(l)) if l % 2 == 0 else '' for l in _data.get_re_axis() / 1e3])
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.set_ylim(-0.5, len(_data.get_re_axis()) - 0.5)
        ax.set_yticks([])
        ax.set_yticklabels([])
    # if _x_ax:
    ax.set_xlim(-0.5, len(_data.get_dt_axis()) - 0.5)
    ax.set_xlim(-0.5, len(_data.get_dt_axis()) - 0.5)
    # ax.set_xticklabels(["{0:1.2f}".format(gmt_anomaly_0 + d / _sst_gmt_factor) for d in _data.get_dt_axis()])
    ax.set_xticks(np.arange(0, len(_data.get_dt_axis()), 1 / _data.dT_stepwidth * _sst_gmt_factor))
    ax.set_xticklabels(np.arange(0, (len(_data.get_dt_axis()) - 1) * _data.dT_stepwidth / _sst_gmt_factor + 1e-10).astype(int))
    if _numbering is not None:
        transform = transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
        fig.text(0, 1, _numbering, ha='left', va='center', fontweight='bold', transform=transform)
    return fig, ax, cbar_ax


def make_heatmap(_data: AggrData, _gauss_filter=False,
                 _gauss_sigma=1, _gauss_truncate=1, _outfile=None, _slopes=None, _ylabel=None, _xlabel=None,
                 _sst_gmt_factor=0.5, _data_division=1.0, _numbering=None, _vmin=None, _vmax=None, _slope_data=None,
                 _y_ax_precision=3, store_data=None):
    if _data.shape[:3] != (1, 1, 1) or _data.shape[-1] != 1 or (
            _slope_data is not None and (_slope_data.shape[:3] != (1, 1, 1) or _slope_data.shape[-1] != 1)):
        raise ValueError("All dimensions of the datasets except lambda and duration.")
    fig, ax, cbar_ax = prepare_heatmap_figure(_data, _type='heatmap', _x_ax=True, _xlabel=_xlabel is not None,
                                              _sst_gmt_factor=_sst_gmt_factor, _numbering=_numbering,
                                              _y_ax_precision=_y_ax_precision)
    _data_aggregated = copy.deepcopy(_data)
    data_array = _data.data.reshape((len(_data.get_re_axis()), len(_data.get_dt_axis())))
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
    cmap = cm.get_cmap('Oranges')
    im = ax.imshow(plot_data, origin='lower', aspect='auto', norm=norm, cmap=cmap)
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar_ax.yaxis.set_ticks_position('left')
    cbar_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.{}f'.format(_y_ax_precision)))
    cbar_ax.yaxis.set_major_locator(ticker.MaxNLocator(8))
    base_x = np.where(_data_aggregated.get_dt_axis() == 0)[0][0]
    base_y = np.where(_data_aggregated.get_re_axis() == 0)[0][0]
    if _slopes is None and _slope_data is not None:
        _slopes = [s * _sst_gmt_factor for s in _slope_data.slope_meta.keys()]
    if _slopes is not None:
        _slopes = sorted(_slopes, reverse=False)
        d_i = (_data_aggregated.get_re_axis()[-1] - _data_aggregated.get_re_axis()[0]) / (len(
            _data_aggregated.get_re_axis()) - 1)
        d_d = (_data_aggregated.get_dt_axis()[-1] - _data_aggregated.get_dt_axis()[0]) / (len(
            _data_aggregated.get_dt_axis()) - 1)
        print(d_i, d_d)
        for idx, slope in enumerate(_slopes):
            m = slope * d_d / d_i / _sst_gmt_factor  # slopes are in km/degC GMT change -> translate into km/degC SST change
            b = base_y - m * base_x
            x_max = min(len(_data_aggregated.get_dt_axis()) - 1,
                        ((len(_data_aggregated.get_re_axis()) - 1) - b) / m)
            y_max = m * x_max + b
            ax.plot([base_x, x_max], [m * base_x + b, y_max], c='w')
            ax.text(base_x + (x_max - base_x) / 1.5, m * (base_x + (x_max - base_x) / 1.5) + b, string.ascii_uppercase[idx],
                    c='w', va='bottom', ha='right')
    if _slope_data is not None:
        binstep_re = abs(_slope_data.get_re()[1] - _slope_data.get_re()[0])
        binstep_dT = abs(_slope_data.get_dt()[1] - _slope_data.get_dt()[0])
        datastep_re = abs(_data.get_re()[1] - _data.get_re()[0])
        datastep_dT = abs(_data.get_dt()[1] - _data.get_dt()[0])
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
            d = np.round(d, 2)
            l = int(l)
            x = d / datastep_dT
            y = l / datastep_re
            z = _slope_data.get_dt(d).get_re(l).data.flatten()[0]
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
    if _ylabel is not None:
        for line_idx, line in enumerate(_ylabel.split('\n')):
            fig.text(0 + 0.04 * line_idx, 0.5, line, rotation=90, ha='left', va='center', transform=trans)
    if _xlabel is not None:
        ax.set_xlabel(_xlabel)
    fig.text(1, 0.5, 'Radius change (km)', rotation=90, ha='right', va='center', transform=trans)
    sector_name = list(_data.get_sectors().keys())[0]
    if sector_name in ['PRIVSECTORS-MINQ', 'ALL_INDUSTRY-MINQ']:
        sector_name = 'All Industry without MQ'
    elif sector_name == 'MINQ':
        sector_name = 'Mining and Quarrying'
    transform = transforms.blended_transform_factory(ax.transAxes, fig.transFigure)
    ax.text(0.5, 0.98, sector_name, transform=transform, ha='center', va='top', fontsize=FSIZE_MEDIUM)
    if _outfile is not None:
        plt.savefig(_outfile, dpi=300)
    if store_data is not None:
        store_slope_data = pd.DataFrame(
            data=_slope_data.data.reshape((21, 15)),
            index=_slope_data.get_re(),
            columns=np.round(_slope_data.get_dt() / _sst_gmt_factor, 2)
        ).unstack().dropna()
        store_main_data = pd.DataFrame(
            data=plot_data,
            index=_data_aggregated.get_re(),
            columns=np.round(_data_aggregated.get_dt() / _sst_gmt_factor, 2)
        ).unstack()
        pd.concat([
            store_slope_data,
            store_main_data.drop(np.intersect1d(store_slope_data.index, store_main_data.index))
        ]).reset_index().rename({'level_0': 'delta_t', 'level_1': 'delta_r', 0: 'value'}, axis=1).to_csv(store_data)
    plt.show(block=False)


def make_heatmap_cut(_data: AggrData, _slopes=None, _outfile=None, _gauss_filter=True, _gauss_sigma=1,
                     _gauss_truncate=1, _plot_xax=True, _ylabel=None, _xlabel=None, _sst_gmt_factor=0.5,
                     _duration_0=0, _numbering=None, _slope_data=None, _y_ax_precision=3, _legend=True,
                     store_data=None):
    if _data.shape[:3] != (1, 1, 1) or _data.shape[-1] != 1 or (
            _slope_data is not None and (_slope_data.shape[:3] != (1, 1, 1) or _slope_data.shape[-1] != 1)):
        raise ValueError("All dimensions of the datasets except lambda and duration.")
    if _slopes is None and _slope_data is None:
        raise ValueError("Must pass either slope array or slope data")
    fig, ax, _ = prepare_heatmap_figure(_data, _type='heatmap_cut', _x_ax=_plot_xax, _sst_gmt_factor=_sst_gmt_factor,
                                        _numbering=_numbering, _y_ax_precision=_y_ax_precision,
                                        _xlabel=_xlabel is not None)
    if _slope_data is None:
        data_array = _data.data.reshape((len(_data.get_re_axis()), len(_data.get_dt_axis())))
        if _slopes is None and _slope_data is not None:
            _slopes = [s * _sst_gmt_factor for s in _slope_data.slope_meta.keys()]
        if _gauss_filter:
            data_array = gaussian_filter(data_array, sigma=_gauss_sigma, mode='nearest', truncate=_gauss_truncate)
        interp = RectBivariateSpline(_data.get_re_axis(), _data.get_dt_axis(), data_array, s=0)
        for idx, slope in enumerate(_slopes):
            durations = _data.get_dt_axis()[_data.get_dt_axis() >= _duration_0]
            x_offset = len(_data.get_dt_axis()[_data.get_dt_axis() < _duration_0])
            lambdas = durations * slope / _sst_gmt_factor  # slopes are in km/degC GMT change. Therefore, translate into km/degC SST change:
            lambdas = lambdas[lambdas <= _data.get_re_axis()[-1]]
            durations = durations[:len(lambdas)]
            z = [interp(lambdas[i], durations[i])[0, 0] for i in range(len(durations))]
            ax.plot(np.arange(len(durations)) + x_offset, z, label=string.ascii_uppercase[idx], linewidth=0.5)
    else:
        datastep_re = abs(_data.get_re()[1] - _data.get_re()[0])
        datastep_dT = abs(_data.get_dt()[1] - _data.get_dt()[0])
        num_slopes = len(_slope_data.slope_meta)
        slope_markers = ['s', 'o', '<', '>', 'x', 'D']
        slope_datapoints = get_slope_means_and_errors(_slope_data)
        for (idx, slope), marker in zip(enumerate(sorted(list(slope_datapoints.keys()), reverse=False)), slope_markers[:num_slopes]):
            mean_vals = slope_datapoints[slope]['mean_vals']
            yerrors = slope_datapoints[slope]['yerrors']
            x_vals = slope_datapoints[slope]['x_vals']
            x_vals = [x / datastep_dT - 2. * datastep_dT * (num_slopes / 2 - 0.5 - idx) for x in x_vals]
            ax.errorbar(x_vals, mean_vals, yerr=yerrors, fmt='o', label=string.ascii_uppercase[idx], linewidth=1,
                        markersize=3, color=REGION_COLORS(idx))  # , capsize=1)
    if _ylabel is not None:
        trans = transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
        for line_idx, line in enumerate(_ylabel.split('\n')):
            fig.text(0 + 0.04 * line_idx, 0.5, line, rotation=90, ha='left', va='center', transform=trans)
    if _xlabel is not None:
        ax.set_xlabel(_xlabel)
    if _legend:
        ax.legend(frameon=False)
    if _outfile is not None:
        plt.savefig(_outfile, dpi=300)
    if store_data is not None:
        data_output = pd.DataFrame(
            columns=['scenario', 'delta_t', 'min', 'mean', 'max']
        )
        for scenario, (_, scenario_data) in enumerate(slope_datapoints.items()):
            for idx, delta_t in enumerate(scenario_data['x_vals']):
                data_output.loc[len(data_output)] = [
                    string.ascii_uppercase[scenario],
                    int(delta_t / _sst_gmt_factor),
                    scenario_data['mean_vals'][idx] - scenario_data['yerrors'][0][idx],
                    scenario_data['mean_vals'][idx],
                    scenario_data['mean_vals'][idx] + scenario_data['yerrors'][1][idx]
                ]
        data_output.to_csv(store_data)
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
                d = np.round(d, 4)
                l = int(l)
                subpoint_val = _slope_data.get_re(l).get_dt(d).data.flatten()[0]
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
    duration_axis = _data.get_dt_axis()
    lambda_axis = _data.get_re_axis()
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
        relative_dev_percent = _data.get_vars(deviation_var).get_re(
            scaled_l).get_dt(scaled_d).get_data().flatten()
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
    if len(_data.get_regions()) != 1:
        raise ValueError('Must pass data with only one region')
    if len(_data.get_sectors()) != 1:
        raise ValueError('Must pass data with only one sector')
    if _x_dim == _y_dim or _y_dim == _z_dim or _z_dim == _x_dim:
        raise ValueError('all dimensions must be different.')
    figsize = (MAX_FIG_WIDTH_NARROW, 0.9 * MAX_FIG_WIDTH_NARROW)
    _data = _data.clip(_clip)
    data_xyz = [[], [], []]
    for _d_idx, _d in enumerate(_data.get_dt_axis()):
        for _l_idx, _l in enumerate(_data.get_re_axis()):
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
                        _data.get_vars(variable).get_re(_l).get_dt(_d).get_data().flatten()[0])
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


def make_agent_var_global_map(_data, _sector='MINQ', _variable='incoming_demand', _dt=0, _re=0, _exclude_regions=None,
                              _t_0=4, _t_agg=365, _plot_shares=True, _cbar_lims=None, _numbering=None, _outfile=None,
                              _symmetric_cbar=True, _show_sector_name=True):
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
    data_array = _data.get_sectors(_sector).get_re(_re).get_dt(_dt).get_data().flatten() * 1e3 / 1e9
    print('Total {} loss: {}'.format(_variable, data_array[data_array < 0].sum()))
    print('Total {} gains: {}'.format(_variable, data_array[data_array >= 0].sum()))
    if _plot_shares:
        data_array[data_array < 0] = data_array[data_array < 0] / abs(sum(data_array[(data_array < 0) & ~data_array.mask]))
        data_array[data_array >= 0] = data_array[data_array >= 0] / abs(
            sum(data_array[(data_array >= 0) & ~data_array.mask]))
        data_array = data_array * 100
    for r, d in sorted(list(zip(_data.get_regions(), data_array)), key=lambda x: x[1]):
        print(r, '{0:1.3f}'.format(d))
    if _cbar_lims is None:
        _cbar_lims = [data_array.min(), data_array.max()]
    plot_data = copy.deepcopy(data_array)
    if _symmetric_cbar:
        positive_scale_factor = abs(_cbar_lims[0]) / _cbar_lims[1]
        plot_data[plot_data >= 0] = plot_data[plot_data >= 0] * positive_scale_factor
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
             data=plot_data,
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
    if _plot_shares:
        fig.text(0.98, 0.5, 'Share (%) of global', ha='right', va='center', rotation=90, transform=trans)
        fig.text(1, 0.5, r'losses   $\longleftrightarrow$   gains', ha='right', va='center', rotation=90,
                 transform=trans)
    else:
        fig.text(0.98, 0.5, r'losses   $\longleftrightarrow$   gains', ha='right', va='center', rotation=90,
                 transform=trans)
        fig.text(1, 0.5, '(bn USD)', ha='right', va='center', rotation=90, transform=trans)
    if _show_sector_name:
        sector_name = _sector
        if sector_name in ['PRIVSECTORS-MINQ', 'ALL_INDUSTRY-MINQ']:
            sector_name = 'All Industry without Mining and Quarrying'
        elif sector_name == 'MINQ':
            sector_name = 'Mining and Quarrying'
        fig.text(0.5, 0.1, sector_name, ha='center', va='center', fontsize=FSIZE_MEDIUM)
    if _numbering is not None:
        fig.text(0, min(ax.get_position().y1, 0.98), _numbering, fontweight='bold', ha='left', va='center')
    if _outfile is not None:
        plt.savefig(_outfile, dpi=300)
    return pd.Series(data_array, _data.get_regions(), name='value')


def generate_flow_dataset(_t0=4, _t_agg=365, _sector='MINQ'):
    data_dir = "/home/robin/repos/harvey_scaling/data/acclimate_output/HARVEY_econYear2015_dT_0_2.5_0.125__re0_100000.0_5000.0__ccFactor1.07/2021-08-11__flow_output/"
    in_data = Dataset(data_dir + "HARVEY_dT0.00_re0_output_with_flows.nc")
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
    data_dir = "/home/robin/repos/harvey_scaling/data/acclimate_output/main_analysis/HARVEY_econYear2015_dT_0_2.5_0.125__re0_100000.0_5000.0__ccFactor1.07/2021-08-11__flow_output/"
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
    subregions = [sr for sr in list(set(WORLD_REGIONS[_region_group]) - set(WORLD_REGIONS.keys())) if sr in _data.get_regions()]
    data_array = _data.get_regions(subregions).data / _ylabel_divisor
    data_array = data_array.reshape((len(subregions), len(_data.get_re_axis()), len(_data.get_dt_axis())))
    data_gains = copy.deepcopy(data_array)
    data_losses = copy.deepcopy(data_array)
    data_gains[data_gains < 0] = 0
    data_losses[data_losses > 0] = 0
    data_gains = data_gains.sum(axis=0)
    data_losses = data_losses.sum(axis=0)
    if _gauss_filter:
        data_gains = gaussian_filter(data_gains, sigma=_gauss_sigma, mode='nearest', truncate=_gauss_truncate)
        data_losses = gaussian_filter(data_losses, sigma=_gauss_sigma, mode='nearest', truncate=_gauss_truncate)
    interp_gains = RectBivariateSpline(_data.get_re_axis(), _data.get_dt_axis(), data_gains, s=0)
    interp_losses = RectBivariateSpline(_data.get_re_axis(), _data.get_dt_axis(), data_losses, s=0)
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
    ax2.set_xticklabels([])
    dT_sst_list = _data.get_dt_axis()
    for idx, slope in enumerate(_slopes):
        dT_gmt_list = dT_sst_list / _sst_gmt_factor
        re_list = dT_gmt_list * slope  # slopes are in km/degC GMT change. Therefore, translate into km/degC SST change:
        re_list = re_list[re_list <= _data.get_re_axis()[-1]]
        global_losses = np.array(
            [regional_data[_region_group]['interp_losses'](re_list[i], dT_sst_list[i])[0, 0] for i in
             range(len(re_list))])
        global_gains = np.array([regional_data[_region_group]['interp_gains'](re_list[i], dT_sst_list[i])[0, 0] for i in
                                 range(len(re_list))])
        dT_gmt_list = dT_gmt_list[:len(re_list)]
        ax.fill_between(dT_gmt_list, global_gains, color='orange', alpha=0.1, linewidth=0)
        ax.fill_between(dT_gmt_list, global_losses, color='purple', alpha=0.1, linewidth=0)
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
        unit_label = '(bn USD)'
    elif _ylabel_divisor == 1e3:
        unit_label = '(m USD)'
    elif _ylabel_divisor == 1:
        unit_label = '(tn USD)'
    ax.text(0.10, 0.05, unit_label, rotation=90, transform=trans, va='center', fontsize=FSIZE_MEDIUM)
    if _x_label:
        trans = transforms.blended_transform_factory(ax.transAxes, fig.transFigure)
        fig.text(0.5, 0 , r'$\Delta T$ (temperature change in °C)', ha='center', va='bottom', transform=trans)
    else:
        ax.set_xticklabels([])
    if _numbering is not None:
        fig.text(0, 1, _numbering, fontweight='bold', ha='left', va='top')
    if _outfile is not None:
        plt.savefig(_outfile, dpi=300)


def plot_compensation_gap(_data: AggrData, region, _t0=4, _t_agg=365, _sst_gmt_factor=0.5, _legend=True,
                          _numbering=None, _xlabel=True, _ylabel=True, _outfile=None, _inplot_region_info=False,
                          _sector_label=True, _remove_upper_tick=False, _ax_right=False, store_data=None):
    if (_data.shape[0], _data.shape[2]) != (1, 1):
        raise ValueError('Can only pass data with one variable and sector.')
    if _data.slope_meta is None:
        raise ValueError('Must pass slope ensemble dataset.')
    _data = _data.get_regions(WORLD_REGIONS['WORLD']).clip(_t0, _t0 + _t_agg).aggregate('absolute_difference')
    _data_gains = copy.deepcopy(_data)
    _data_losses = copy.deepcopy(_data)
    _data_gains.data_capsule.data[_data.data < 0] = 0
    _data_losses.data_capsule.data[_data.data > 0] = 0
    _data_gains.data_capsule.data = np.abs(_data_gains.data)
    _data_losses.data_capsule.data = np.abs(_data_losses.data)
    clean_regions(_data_gains)
    clean_regions(_data_losses)
    _data = _data_gains
    _data.data_capsule.data = (_data_gains.data / _data_losses.data) * 100
    fig_width = MAX_FIG_WIDTH_NARROW
    fig_height = MAX_FIG_WIDTH_NARROW
    y_scale = 1
    if not _xlabel:
        y_scale -= 0.05
    if _sector_label:
        y_scale += 0.02
    x_scale = 1
    fig, ax = plt.subplots(figsize=(fig_width * x_scale, fig_height * y_scale))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.xaxis.set_ticks_position('none')
    ax.tick_params('x', pad=1)
    if _ax_right:
        ax.yaxis.tick_right()
    ax_width = 0.85 / x_scale
    ax_height = 0.85 / y_scale
    ax_x0 = 0.13 / x_scale if not _ax_right else (1 - 0.11) / x_scale - ax_width
    ax_y0 = 0.1 / y_scale if _xlabel else 0.05
    ax.set_position([ax_x0, ax_y0 , ax_width, ax_height])
    num_slopes = len(_data.slope_meta)
    x_stepwidth = (list(_data.slope_meta[0].keys())[0][0] - list(_data.slope_meta[0].keys())[1][0]) / _sst_gmt_factor
    slope_markers = ['s', 'o', '<', '>', 'x', 'D']
    output_data = pd.DataFrame(
        columns=['min', 'mean', 'max'],
        index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=[u'scenario', u'delta_t'])
    )
    for (idx, slope), marker in zip(enumerate(sorted(list(_data.slope_meta.keys()), reverse=False)), slope_markers[:num_slopes]):
        datapoints = get_slope_means_and_errors(_data.get_regions(region))
        mean_vals = datapoints[slope]['mean_vals']
        yerrors = datapoints[slope]['yerrors']
        x_vals = datapoints[slope]['x_vals'] / _sst_gmt_factor
        plot_x_vals = [x + x_stepwidth / (2.5 * num_slopes) * (num_slopes / 2 - 0.5 - idx) for x in x_vals]
        ax.errorbar(plot_x_vals, mean_vals, yerr=yerrors, fmt='o', label=string.ascii_uppercase[idx],
                    linewidth=1, markersize=3, color=REGION_COLORS(idx))
        for t_idx, delta_t in enumerate(x_vals):
            index = (string.ascii_uppercase[idx], int(delta_t))
            output_data.loc[index, :] = [
                mean_vals[t_idx] - yerrors[0][t_idx],
                mean_vals[t_idx],
                mean_vals[t_idx] + yerrors[1][t_idx]
            ]
    if _legend:
        plt.legend(frameon=False)
    if _inplot_region_info:
        sector_name = list(_data.get_sectors().keys())[0]
        if sector_name in ['PRIVSECTORS-MINQ', 'ALL_INDUSTRY-MINQ']:
            sector_name = 'All Industry without MQ'
        elif sector_name == 'MINQ':
            sector_name = 'Mining and Quarrying'
        region_name = region
        if region_name == 'WORLD':
            region_name = 'World'
        text = "{}, {}".format(sector_name, region_name)
        ax.text(0.02, 0.02 / y_scale, text, transform=ax.transAxes, ha='left', va='bottom')
    if _xlabel:
        trans = transforms.blended_transform_factory(ax.transAxes, fig.transFigure)
        fig.text(0.5, 0 / y_scale, r'$\Delta T$ (temperature change in °C)', ha='center', va='bottom', transform=trans,
                 fontsize=FSIZE_MEDIUM)
    if _ylabel:
        trans = transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
        region_name = region
        if region_name == 'WORLD':
            region_name = 'Global'
        fig.text(0, 0.5 / y_scale, '{} CLR'.format(region_name), rotation=90, ha='left',
                 va='center', transform=trans, fontsize=FSIZE_MEDIUM)
    if _remove_upper_tick:
        ax.set_yticks(ax.get_yticks()[:-2])
    if _sector_label:
        sector_name = list(_data.get_sectors().keys())[0]
        if sector_name in ['PRIVSECTORS-MINQ', 'ALL_INDUSTRY-MINQ']:
            sector_name = 'All Industry without MQ'
        elif sector_name == 'MINQ':
            sector_name = 'Mining and Quarrying'
        transform = transforms.blended_transform_factory(ax.transAxes, fig.transFigure)
        fig.text(0.5, 0.98, sector_name, ha='center', va='top', transform=transform, fontsize=FSIZE_MEDIUM)
    if _numbering is not None:
        trans = transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
        fig.text(0.05 if not _ax_right else 0, 1, _numbering, fontweight='bold', ha='left', va='top', transform=trans)
    if _outfile is not None:
        plt.savefig(_outfile, dpi=300)
    if store_data is not None:
        output_data.to_csv(store_data)


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
        data_gains = data_gains[0, :, 0, :, :, 0]
        data_gains[data_gains < 0] = 0
        data_gains = data_gains.sum(axis=0)
        if _gauss_filter is not None:
            data_gains = gaussian_filter(data_gains, sigma=_gauss_sigma, mode='nearest', truncate=_gauss_truncate)
        interp_gains = RectBivariateSpline(_data.get_re_axis(), _data.get_dt_axis(), data_gains, s=0)
        regional_data[region_name] = {
            'gains': data_gains,
            'interp_gains': interp_gains
        }
    fig, axs = plt.subplots(int(np.sqrt(len(_region_groups))), int(np.ceil(np.sqrt(len(_region_groups)))))
    dT_sst_list = _data.get_dt_axis()
    for region_group, ax in zip(_region_groups, axs.flatten()):
        ax.set_title(region_group)
        for idx, slope in enumerate(_slopes):
            dT_gmt_list = dT_sst_list / _sst_gmt_factor
            re_list = dT_gmt_list * slope  # slopes are in km/degC GMT change. Therefore, translate into km/degC SST change:
            re_list = re_list[re_list <= _data.get_re_axis()[-1]]
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
                                          _numbering=None, _relative=True, _t0=4, _t_agg=365,
                                          _ylim=None, _ylabel=True, _xlabel=True, _ax_right=False, _bar_reg_label=True,
                                          _inplot_legend=True, _inplot_sector_label=True,
                                          _minor_ytick_loc=None, _major_ytick_loc=None, _plot_regression=None,
                                          store_data=None):
    if (_data.shape[0], _data.shape[2]) != (1, 1):
        raise ValueError('Can only pass data with one variable and sector.')
    if type(_region_groups) == str:
        _region_groups = [_region_groups]
    _data = _data.get_regions(list(set(WORLD_REGIONS['WORLD']) - {'US.TX', 'US.LA'})).clip(_t0, _t0 + _t_agg).aggregate('absolute_difference')
    clean_regions(_data)
    if _relative:
        _data.data_capsule.data = _data.data / _data.get_regions('WORLD').data * 100
    else:
        _data.data_capsule.data = _data.data / 1e6
    y_scale = 1
    if not _xlabel:
        y_scale -= 0.07
    if _inplot_sector_label:
        y_scale += 0.05
    x_scale = 1
    if _ax_right and not _ylabel:
        x_scale -= .02
    fig, ax = plt.subplots(figsize=(MAX_FIG_WIDTH_NARROW * x_scale, MAX_FIG_WIDTH_NARROW * y_scale))
    ax_width = 0.85 / x_scale
    ax_height = 0.85 / y_scale
    ax_x0 = 0.12 / x_scale if not _ax_right else (1 - 0.12) / x_scale - ax_width
    ax_y0 = 0.11 / y_scale if _xlabel else 0.02 / y_scale
    ax.set_position([ax_x0, ax_y0, ax_width, ax_height])
    if _ax_right:
        ax.yaxis.tick_right()
    num_slopes = len(_data.slope_meta.keys())
    num_regions = len(_region_groups)
    bar_width = 1
    minor_xtick_positions = np.array([])
    minor_xtick_labels = np.array([])
    major_xtick_positions = []
    if _minor_ytick_loc is not None:
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(_minor_ytick_loc))
    if _major_ytick_loc is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(_major_ytick_loc))
    output_data = pd.DataFrame(
        columns=['min', 'mean', 'max'],
        index=pd.MultiIndex(levels=[[], [], []], codes=[[], [], []], names=[u'scenario', u'delta_t', u'region_group'])
    )
    for slope_idx, slope in enumerate(sorted(list(_data.slope_meta.keys()), reverse=False)):
        pre_mean_vals = None
        for region_idx, region_group in enumerate(_region_groups):
            slope_datapoints = get_slope_means_and_errors(_data.get_regions(region_group))
            mean_vals = slope_datapoints[slope]['mean_vals']
            yerrors = slope_datapoints[slope]['yerrors']
            x_vals = (slope_datapoints[slope]['x_vals'] / _sst_gmt_factor)
            x_positions = x_vals + (len(mean_vals) + 0.5) * slope_idx
            if region_idx == 0:
                minor_xtick_positions = np.concatenate((minor_xtick_positions, x_positions))
                minor_xtick_labels = np.concatenate((minor_xtick_labels, x_vals))
                major_xtick_positions.append(np.mean(x_positions))
            if pre_mean_vals is not None:
                mean_delta = mean_vals - pre_mean_vals
            else:
                mean_delta = mean_vals
            ax.bar(x_positions, mean_delta, width=bar_width, yerr=yerrors, label=slope, align='center', bottom=pre_mean_vals,
                   color=REGION_COLORS(slope_idx), alpha=(1 - region_idx * 0.2), error_kw={'alpha': 0.3})
            if slope_idx == 0:
                ax.axhline(mean_delta[0] + (pre_mean_vals[0] if pre_mean_vals is not None else 0), c='k', linewidth=0.5,
                           xmin=0.02, xmax=0.98)
            if _plot_regression is not None:
                if region_group in _plot_regression.keys():
                    if string.ascii_uppercase[slope_idx] in _plot_regression[region_group]:
                        reg = stats.linregress(x_positions, mean_vals)
                        x_fit = [x_positions[0], x_positions[-1]]
                        y_fit = [reg.intercept + x_positions[0] * reg.slope, reg.intercept + x_positions[-1] * reg.slope]
                        ax.plot(x_fit, y_fit, color='b')#, linestyle='--')
            if _inplot_legend:
                rectangle_height = 0.05
                rectangle_width = 0.05
                x = 0.05 + slope_idx * rectangle_width
                y = 0.95 - (num_regions - region_idx) * rectangle_height
                rect = patches.Rectangle((x, y), rectangle_width, rectangle_height, edgecolor=None, linewidth=0,
                                         facecolor=REGION_COLORS(slope_idx), alpha=(1 - region_idx * 0.2),
                                         transform=ax.transAxes)
                ax.add_patch(rect)
                if slope_idx == num_slopes - 1:
                    ax.text(x + 0.06, y + rectangle_height / 2, region_group.replace('MINQ', 'MQ'), ha='left',
                            va='center', transform=ax.transAxes)
                if region_idx == 0:
                    ax.text(x + rectangle_width / 2, y - 0.01, string.ascii_uppercase[slope_idx], ha='center', va='top',
                            transform=ax.transAxes)
            pre_mean_vals = mean_vals
            for idx, delta_t in enumerate(x_vals):
                index = (string.ascii_uppercase[slope_idx], int(delta_t), region_group)
                output_data.loc[index, :] = [
                    mean_vals[idx] - yerrors[0][idx],
                    mean_vals[idx],
                    mean_vals[idx] + yerrors[1][idx]
                ]
    if _inplot_sector_label:
        sector_name = list(_data.get_sectors().keys())[0]
        if sector_name in ['PRIVSECTORS-MINQ', 'ALL_INDUSTRY-MINQ']:
            sector_name = 'All Industry without MQ'
        elif sector_name == 'MINQ':
            sector_name = 'Mining and Quarrying'
        transform = transforms.blended_transform_factory(ax.transAxes, fig.transFigure)
        fig.text(0.5, 0.98, sector_name, ha='center', va='top', transform=transform, fontsize=FSIZE_MEDIUM)
    if _ylabel:
        if _relative:
            label = 'Share of global gains (%)'
        else:
            label = 'Absolute gains (bn USD)'
        transform = transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
        fig.text(0, .5, label, ha='left', va='center', rotation=90, transform=transform, fontsize=FSIZE_MEDIUM)
    ax.set_xticks(minor_xtick_positions)
    if _xlabel:
        ax.set_xticklabels([int(l) for l in minor_xtick_labels])
        transform = transforms.blended_transform_factory(ax.transAxes, fig.transFigure)
        fig.text(0.5, 0, r'$\Delta T$ (temperature change in °C)', ha='center', va='bottom', transform=transform,
                 fontsize=FSIZE_MEDIUM)
    else:
        ax.set_xticklabels([])
    if _ylim is not None:
        ax.set_ylim(_ylim)
    if _numbering is not None:
        transform = transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
        fig.text(0.05 if not _ax_right else 0, 1, _numbering, ha='left', va='top', fontweight='bold', transform=transform)
    if _outfile is not None:
        plt.savefig(_outfile, dpi=300)
    if store_data is not None:
        output_data.to_csv(store_data)
    return _data


def plot_gains_and_losses_from_slope_dataset(_slope_data: AggrData, _region_group, _data: AggrData = None, _sst_gmt_factor=0.5, _outfile=None,
                                             _numbering=None, _relative=True, _t0=4, _t_agg=365, _ax_right=False,
                                             _ylim=None, _ylabel=True, _xlabel=True, _sector_label=False,
                                             _upper_slope_labels=False, _minor_ytick_loc=None, _major_ytick_loc=None,
                                             store_data=None):
    if (_slope_data.shape[0], _slope_data.shape[2]) != (1, 1):
        raise ValueError('Can only pass data with one variable and sector.')
    if _data is not None and (_data.shape[0], _data.shape[2]) != (1, 1):
        raise ValueError('Can only pass data with one variable and sector.')
    if _data is not None:
        _data = _data.get_regions(WORLD_REGIONS['WORLD']).clip(_t0, _t0 + _t_agg).aggregate('absolute_difference')
        _data.data_capsule.data = _data.data / 1e6
        clean_regions(_data)
    _slope_data = _slope_data.get_regions(WORLD_REGIONS['WORLD']).clip(_t0, _t0 + _t_agg).aggregate('absolute_difference')
    _slope_data.data_capsule.data = _slope_data.data / 1e6
    slopedata_gains = copy.deepcopy(_slope_data)
    slopedata_gains.data[slopedata_gains.data < 0] = 0
    slopedata_losses = copy.deepcopy(_slope_data)
    slopedata_losses.data[slopedata_losses.data > 0] = 0
    clean_regions(_slope_data)
    clean_regions(slopedata_gains)
    clean_regions(slopedata_losses)
    y_scale = .94
    if not _xlabel:
        y_scale -= 0.07
    if _sector_label:
        y_scale += 0.05
    x_scale = 1
    if _ax_right and not _ylabel:
        x_scale -= .07
    fig, ax = plt.subplots(figsize=(MAX_FIG_WIDTH_NARROW * x_scale, MAX_FIG_WIDTH_NARROW * y_scale))
    ax_width = 0.8 / x_scale
    ax_height = 0.8 / y_scale
    ax_x0 = 0.18 / x_scale if not _ax_right else (1 - 0.16) / x_scale - ax_width
    ax_y0 = 0.12 / y_scale if _xlabel else 0.02 / y_scale
    ax.set_position([ax_x0, ax_y0, ax_width, ax_height])
    num_slopes = len(_slope_data.slope_meta.keys())
    bar_width = 1
    if _ax_right:
        ax.yaxis.tick_right()
    xtick_positions = np.array([])
    xtick_labels = np.array([])
    slope_xtick_positions = []
    if _minor_ytick_loc is not None:
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(_minor_ytick_loc))
    if _major_ytick_loc is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(_major_ytick_loc))
    ax2 = ax.twiny()
    ax2.set_position([ax_x0, ax_y0, ax_width, ax_height])
    ax2.spines['top'].set_position(('data', 0))
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.set_xticklabels([])
    output_data = pd.DataFrame(
        columns=['gains_min', 'gains_mean', 'gains_max', 'losses_min', 'losses_mean', 'losses_max', 'sum_min',
                 'sum_mean', 'sum_max'],
        index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=[u'scenario', u'delta_t'])
    )
    for slope_idx, slope in enumerate(sorted(list(_slope_data.slope_meta.keys()), reverse=False)):
        for dataset, color, col_name in zip([slopedata_gains, slopedata_losses, _slope_data],
                                            ['orange', 'purple', 'dimgrey'],
                                            ['gains', 'losses', 'sum']):
            slope_datapoints = get_slope_means_and_errors(dataset.get_regions(_region_group))
            mean_vals = slope_datapoints[slope]['mean_vals']
            yerrors = slope_datapoints[slope]['yerrors']
            x_vals = (slope_datapoints[slope]['x_vals'] / _sst_gmt_factor)
            x_positions = x_vals + (len(mean_vals) + 0.5) * slope_idx
            if dataset == slopedata_gains:
                xtick_positions = np.concatenate((xtick_positions, x_positions))
                xtick_labels = np.concatenate((xtick_labels, x_vals))
                slope_xtick_positions.append(np.mean(x_positions))
            ax.bar(x_positions, mean_vals, width=bar_width, label=slope, align='center', color=color, alpha=0.5, zorder=1)
            ax.bar(x_positions, mean_vals, width=0, yerr=yerrors, label=slope, align='center', color='none',
                   alpha=0.5, zorder=3)
            for idx, delta_t in enumerate(x_vals):
                index = (string.ascii_uppercase[slope_idx], np.round(delta_t, 2))
                output_data.loc[index, [f"{col_name}_min", f"{col_name}_mean", f"{col_name}_max"]] = [
                    mean_vals[idx] - yerrors[0][idx],
                    mean_vals[idx],
                    mean_vals[idx] + yerrors[1][idx],
                ]
        if _data is not None:
            subregions = [r for r in list(set(WORLD_REGIONS[_region_group]) - set(WORLD_REGIONS.keys())) if r in _data.get_regions()]
            data_array = _data.get_regions(subregions).data
            data_array = data_array.reshape((len(subregions), len(_data.get_re_axis()), len(_data.get_dt_axis())))
            data_gains = copy.deepcopy(data_array)
            data_losses = copy.deepcopy(data_array)
            data_gains[data_gains < 0] = 0
            data_losses[data_losses > 0] = 0
            data_gains = np.nansum(data_gains, axis=0)
            data_losses = np.nansum(data_losses, axis=0)
            data_sum = np.nansum(data_array, axis=0)
            dT_sst_list = _data.get_dt_axis()
            dT_gmt_list = dT_sst_list / _sst_gmt_factor
            re_list = dT_sst_list * slope
            re_list = re_list[re_list <= _data.get_re_axis()[-1]]
            for dataset, color, col_name in zip([data_gains, data_losses, data_sum], ['orange', 'purple', 'dimgrey'],
                                                ['gains', 'losses', 'sum']):
                interp = RectBivariateSpline(_data.get_re_axis(), _data.get_dt_axis(), dataset, s=0)
                plot_data = np.array([interp(re_list[i], dT_sst_list[i])[0, 0] for i in range(len(re_list))])
                dT_gmt_list = dT_gmt_list[:len(re_list)]
                ax.plot(dT_gmt_list + (max(dT_gmt_list) + 1.5) * slope_idx, plot_data, color=color, alpha=1, zorder=2)
                for idx, delta_t in enumerate(dT_gmt_list):
                    index = (string.ascii_uppercase[slope_idx], np.round(delta_t, 2))
                    output_data.loc[index, f"{col_name}_mean"] = plot_data[idx]
    if _sector_label:
        sector_name = list(_slope_data.get_sectors().keys())[0]
        if sector_name in ['PRIVSECTORS-MINQ', 'ALL_INDUSTRY-MINQ']:
            sector_name = 'All Industry without MQ'
        elif sector_name == 'MINQ':
            sector_name = 'Mining and Quarrying'
        transform = transforms.blended_transform_factory(ax.transAxes, fig.transFigure)
        fig.text(0.5, 0.98, sector_name, ha='center', va='top', transform=transform, fontsize=FSIZE_MEDIUM)
    if _ylabel:
        transform = transforms.blended_transform_factory(fig.transFigure, ax.transData)
        fig.text(0, 0, r'losses   $\longleftrightarrow$   gains', ha='left', va='center', rotation=90,
                 transform=transform, fontsize=FSIZE_MEDIUM)
        fig.text(0.055, 0, '(bn USD)', ha='left', va='center', rotation=90, transform=transform, fontsize=FSIZE_MEDIUM)
    ax.set_xticks(xtick_positions)
    if _xlabel:
        ax.set_xticklabels([int(l) for l in xtick_labels])
        transform = transforms.blended_transform_factory(ax.transAxes, fig.transFigure)
        fig.text(0.5, 0, r'$\Delta T$ (temperature change in °C)', ha='center', va='bottom', transform=transform,
                 fontsize=FSIZE_MEDIUM)
    else:
        ax.set_xticklabels([])
    transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for pos, slope_label in zip(slope_xtick_positions, string.ascii_uppercase[:num_slopes]):
        fig.text(pos, 0.025 if not _upper_slope_labels else 0.97, slope_label, ha='center', transform=transform,
                 va='bottom' if not _upper_slope_labels else 'top')
    if _ylim is not None:
        ax.set_ylim(_ylim)
    if _numbering is not None:
        transform = transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
        fig.text(0.06 if not _ax_right else 0, 1, _numbering, ha='left', va='top', fontweight='bold',
                 transform=transform)
    if _outfile is not None:
        plt.savefig(_outfile, dpi=300)
    if store_data is not None:
        output_data.to_csv(store_data)


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
        sector_gain_shares = sector_gain_shares.reshape((len(_data.get_re_axis()), len(_data.get_dt_axis())))
        interp_gain_shares = RectBivariateSpline(_data.get_re_axis(), _data.get_dt_axis(), sector_gain_shares,
                                                 s=0)
        regional_data[region_name] = {
            'gain_shares': sector_gain_shares,
            'interp_gain_shares': interp_gain_shares
        }
    fig, axs = plt.subplots(int(np.ceil(np.sqrt(len(_regions)))), int(np.ceil(np.sqrt(len(_regions)))))
    dT_sst_list = _data.get_dt_axis()
    for region_group, ax in zip(_regions, axs.flatten()):
        ax.set_title(region_group)
        for idx, slope in enumerate(_slopes):
            dT_gmt_list = dT_sst_list / _sst_gmt_factor
            re_list = dT_gmt_list * slope  # slopes are in km/degC GMT change. Therefore, translate into km/degC SST change:
            re_list = re_list[re_list <= _data.get_re_axis()[-1]]
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
        mode='absolute', _aggregate=True).get_sectors(_sector).get_re(_re).get_dt(_dt)
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


def make_receipt_output(_data: AggrData, _slope_data: AggrData, _sst_gmt_ratio=0.5, _pdf_output=False):
    temperatures = [0, 1, 2, 3, 4, 5]
    slopes = [10000]
    outfile_path = "/home/robin/repos/harvey_scaling/RECEIPT/reduced_data/"

    production_data = None
    for dt_global in temperatures:
        for slope in slopes:
            dt = dt_global * _sst_gmt_ratio
            re = dt_global * slope
            if _pdf_output:
                outfile = outfile_path + "map_plots/map_{}_{}.pdf".format(dt / _sst_gmt_ratio, re)
            else:
                outfile = None
            prod_anomaly = make_agent_var_global_map(_data, _sector='PRIVSECTORS', _variable='production', _dt=dt, _re=re,
                                                     _cbar_lims=[-40, 4], _symmetric_cbar=True, _plot_shares=False,
                                                     _show_sector_name=False, _outfile=outfile)
            prod_anomaly.loc['radius_extension'] = re
            prod_anomaly.loc['temperature_increase'] = dt / _sst_gmt_ratio
            prod_anomaly = prod_anomaly.transpose().set_index(['temperature_increase', 'radius_extension'], drop=True)
            if production_data is None:
                production_data = prod_anomaly
            else:
                production_data = production_data.append(prod_anomaly)
            plt.close('all')
    production_data.rename(cn_gadm_to_iso_code, axis=1, inplace=True)
    production_data = production_data[sorted(production_data.columns)]
    production_data.index.set_names(['temperature_increase [degC]', 'radius_extension [km]'], inplace=True)
    production_data.to_csv(outfile_path + "production_anomaly.csv")

    plot_radius_extension_map(_outfile=outfile_path + "radius_map/radius_map.pdf", _shape_outpath=outfile_path + "radius_map/",
                              re_selection=[t * re_ratio for re_ratio in slopes for t in temperatures])

    gain_shares = plot_global_shares_from_slope_dataset(_slope_data.get_sectors('PRIVSECTORS'),
                                                        _region_groups=['USA', 'EXPORT:50', 'EXPORT:75', 'EXPORT:95'])
    gain_shares_output = pd.DataFrame(columns=['USA', 'EXPORT:50', 'EXPORT:75', 'EXPORT:95'],
                                      index=pd.MultiIndex.from_tuples(zip(temperatures, [t * slope / 1000 for t in temperatures for slope in slopes])))
    for region in gain_shares_output.columns:
        shares = get_slope_means_and_errors(gain_shares.get_regions(region))
        for t in temperatures:
            for s in slopes:
                t_idx = np.where(shares[s]['x_vals'] == t * _sst_gmt_ratio)[0][0]
                gain_shares_output.loc[(t, t * s / 1000), region] = shares[s]['mean_vals'][t_idx]
    gain_shares_output.iloc[:, 1:] = gain_shares_output.iloc[:, 1:] - gain_shares_output.iloc[:, :-1].values
    gain_shares_output['WORLD'] = 100 - gain_shares_output.sum(axis=1)
    gain_shares_output.index.set_names(['temperature_increase [degC]', 'radius_extension [km]'], inplace=True)
    gain_shares_output.to_csv(outfile_path + "gain_shares.csv")


def sst_gmt_relationship(sea, hurricane_months_only=True, regression=True, monthly_diff_to_historical=True,
                         outfile=None):
    fig_width = MAX_FIG_WIDTH_WIDE
    fig_height = MAX_FIG_WIDTH_WIDE * 0.5
    fig, axs = plt.subplots(ncols=3, sharey=True, sharex=True, figsize=(fig_width, fig_height), constrained_layout=True)
    mask = xr.load_dataarray("/home/robin/data/isimip/masks/generated/{}.nc".format(sea.replace(' ', '_')))
    markers = ['d', '^', 'o', 's', 'p']
    ssp_names = ['SSP-126', 'SSP-370', 'SSP-585']
    scatter_points = {
        'ssp126': [[], []],
        'ssp370': [[], []],
        'ssp585': [[], []]
    }
    for model_idx, model in tqdm.tqdm(enumerate(['GFDL-ESM4', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'UKESM1-0-LL'])):
        historical = xr.load_dataarray("/home/robin/data/isimip/aggregated/monthly/historical/{}.nc".format(model))
        historical = historical.sel(time=slice('1850-01-01', '1899-12-31')).groupby('time.month').mean(dim='time')
        weights = np.cos(np.deg2rad(historical.lat))
        for ssp_idx, ssp in enumerate(['ssp126', 'ssp370', 'ssp585']):
            projection = xr.load_dataarray("/home/robin/data/isimip/aggregated/monthly/{}/{}.nc".format(ssp, model))
            if monthly_diff_to_historical:
                projection = projection.groupby('time.month') - historical
            mean_temp_masked = (projection * mask).weighted(weights).mean(dim=['lon', 'lat'])
            mean_temp_global = projection.weighted(weights).mean(dim=['lon', 'lat'])
            label = model if ssp_idx == 0 else None
            if hurricane_months_only:
                mean_temp_masked = mean_temp_masked[mean_temp_masked.month.isin([6, 7, 8, 9, 10, 11])]
                mean_temp_global = mean_temp_global[mean_temp_global.month.isin([6, 7, 8, 9, 10, 11])]
            axs[ssp_idx].scatter(mean_temp_global.values, mean_temp_masked.values, marker=markers[model_idx],
                                   label=label, facecolors='none', edgecolors='k', s=10, linewidths=0.5, alpha=0.2)
            scatter_points[ssp][0] += list(mean_temp_global.values)
            scatter_points[ssp][1] += list(mean_temp_masked.values)
    if regression:
        for ssp_idx, ssp in enumerate(['ssp126', 'ssp370', 'ssp585']):
            x, y = scatter_points[ssp]
            reg = stats.linregress(x, y)
            axs[ssp_idx].plot([min(x), max(x)], reg.intercept + reg.slope * np.array([min(x), max(x)]), 'r')
            stats_legend = "y = {0:.3f}x + {1:.3f}\nr = {2:.3f}".format(reg.slope, reg.intercept, reg.rvalue)
            axs[ssp_idx].text(0.99, 0.01, stats_legend, ha='right', va='bottom', transform=axs[ssp_idx].transAxes,
                              fontsize=FSIZE_TINY, color='r')
    axs[1].set_xlabel('global mean temperature anomaly [°C]')
    axs[0].set_ylabel('{} temperature anomaly [°C]'.format(sea))
    axs[0].legend()
    plt.tight_layout()
    for ax_idx, ax in enumerate(axs):
        pos_old = ax.get_position()
        height_new = pos_old.height * 0.99
        ax.set_position([pos_old.x0, pos_old.y0, pos_old.width, height_new])
        ax.text(0, 0.99, ssp_names[ax_idx], ha='left', va='top', fontweight='bold', transform=ax.transAxes)
        fig.text(pos_old.x0, 1, chr(ax_idx + 97), ha='center', va='top', fontweight='bold')
    if outfile is not None:
        plt.savefig(outfile, dpi=300)


def plot_sensitivity_analysis(path, name=None, store_data=None, fig_numbers=None):
    t_max = 365
    data = {}
    abs_gains_and_losses = {}
    rel_gains_and_losses = {}
    for folder in tqdm.tqdm(list(os.listdir(path))):
        directory = os.path.join(path, folder)
        if os.path.isdir(directory):
            data[folder] = {}
            abs_gains_and_losses[folder] = {}
            rel_gains_and_losses[folder] = {}
            for subfolder in tqdm.tqdm(list(os.listdir(directory))[:]):
                subdir = os.path.join(directory, subfolder)
                if os.path.isdir(subdir):
                    parameter = float(subfolder)
                    acc_output = AcclimateOutput(os.path.join(subdir, 'output.nc'), old_output_format=True,
                                                 start_date='2000-01-01')
                    acc_output = acc_output[['firms_production', 'firms_consumption']]
                    acc_output = acc_output.sel(agent_type='firm')
                    acc_output._data['net_production'] = (acc_output.firms_production - acc_output.firms_consumption).data
                    acc_output._baseline = acc_output.data.isel(time=0)
                    acc_output = acc_output.isel(time=np.arange(min(t_max, len(acc_output.time.values))))
                    acc_output.group_agents(dim='sector', group=list(acc_output.agent_sector.values),
                                            name='ALL_INDUSTRY', inplace=True)
                    acc_output = acc_output.sel(agent_sector='ALL_INDUSTRY')
                    data[folder][parameter] = acc_output
                    num_days = len(acc_output.time.values)
                    baseline_prod = acc_output.net_production.isel(time=0).data * num_days
                    abs_gl = acc_output.net_production.sum(dim='time').data - baseline_prod
                    rel_gl = abs_gl / baseline_prod
                    abs_gains_and_losses[folder][parameter] = abs_gl
                    rel_gains_and_losses[folder][parameter] = rel_gl

    default_params = {
        'initial_storage_fill_factor': 10,
        'price_increase_production_extension': 5.0,
        'target_storage_refill_time': 2,
    }

    reference_default_run = 'initial_storage_fill_factor'
    for param in ['price_increase_production_extension', 'target_storage_refill_time']:
        data[param][default_params[param]] = data[reference_default_run][default_params[reference_default_run]]
        abs_gains_and_losses[param][default_params[param]] = abs_gains_and_losses[reference_default_run][default_params[reference_default_run]]
        rel_gains_and_losses[param][default_params[param]] = rel_gains_and_losses[reference_default_run][default_params[reference_default_run]]

    WORLD = list(list(data.values())[0].values())[0].agent_region.values
    USA = [r for r in WORLD if r[:3] == 'US.']

    parameter_letters = {
        'initial_storage_fill_factor': r'$\Psi$',
        'price_increase_production_extension': r'$\Delta n^{in,v,>}$',
        'target_storage_refill_time': r'$\tau$',
    }

    fig1_data_output = pd.DataFrame(
        columns=np.arange(t_max),
        index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=[u'parameter', u'param_val'])
    )
    fig1, axs1 = plt.subplots(nrows=len(data.keys()), ncols=1, sharex=True, sharey=True,
                              figsize=(MAX_FIG_WIDTH_WIDE, MAX_FIG_WIDTH_WIDE))
    paramter_colors = ['Green', 'Blue', 'Orange']
    for row_idx, parameter in tqdm.tqdm(zip(range(len(axs1)), data.keys())):
        cmap = plt.cm.get_cmap(paramter_colors[row_idx] + 's')
        for idx, (param_value, dataset) in enumerate(sorted(data[parameter].items())):
            if param_value in [min(data[parameter].keys()), max(data[parameter].keys()), default_params[parameter]]:
                label = int(param_value) if parameter in ['initial_storage_fill_factor'] else param_value
            else:
                label = '_nolegend_'
            color = cmap(.1 + .8 * idx / (len(data[parameter]) - 1))
            linestyle = None
            alpha = 1
            zorder = 0
            if param_value == default_params[parameter]:
                color = 'k'
                linestyle = 'dotted'
                alpha = 0.75
                zorder = 5
            plot_data = (((dataset.net_production.sum(dim='agent') / dataset.net_production.isel(time=0).sum(dim='agent')).data - 1) * 100).values
            axs1[row_idx].plot(plot_data, label=label, color=color, linestyle=linestyle, alpha=alpha, zorder=zorder)
            fig1_data_output.loc[(parameter, param_value), :] = plot_data
        axs1[row_idx].legend(loc='lower right')
    axs1[-1].set_xlabel('time [days]', fontdict={'fontsize': FSIZE_MEDIUM})
    axs1[int(len(axs1) / 2)].set_ylabel('deviation [%]', fontdict={'fontsize': FSIZE_MEDIUM})
    fig1.tight_layout()
    for col, factor in enumerate(np.arange(len(axs1), 0, -1) - 1):
        ax = axs1[col]
        x, y, w, h = ax.get_position().x0, ax.get_position().y0, ax.get_position().width, ax.get_position().height
        shrink = 0.01
        move = 0.005
        ax.set_position([x, y - move * factor, w, h - shrink])
    for ax_idx, ax in enumerate(axs1.flatten()):
        ax.text(0, 1, chr(ax_idx + 97), fontweight='bold', ha='left', va='bottom', transform=ax.transAxes)
        ax.set_xlim((0, 130))
    title_transform = transforms.blended_transform_factory(axs1[0].transAxes, fig1.transFigure)
    fig1.text(0.5, 1, 'Global production anomaly', fontdict={'fontsize': FSIZE_MEDIUM}, ha='center', va='top',
              transform=title_transform)

    fig2_data_output = pd.DataFrame(
        columns=['gains', 'losses', 'sum'],
        index=pd.MultiIndex(levels=[[], [], []], codes=[[], [], []], names=[u'parameter', u'param_val', u'region'])
    )
    col_width = {
        'initial_storage_fill_factor': 1,
        'price_increase_production_extension': 0.5,
        'target_storage_refill_time': 0.5,
    }
    tick_formatters = {
        'initial_storage_fill_factor': (col_width['initial_storage_fill_factor'], 2),
        'price_increase_production_extension': (col_width['price_increase_production_extension'], 1),
        'target_storage_refill_time': (col_width['target_storage_refill_time'], 1),
    }
    fig2, axs2 = plt.subplots(nrows=len(data.keys()), ncols=2, sharex=False, sharey=False,
                              figsize=(MAX_FIG_WIDTH_WIDE, MAX_FIG_WIDTH_WIDE))
    for row_idx, parameter in tqdm.tqdm(list(zip(range(len(axs1)), data.keys()))):
        for param_value, dataset in sorted(list(data[parameter].items())):
            for col_idx, (region_group, region_group_name) in enumerate(zip([WORLD, USA], ['global', 'USA'])):
                abs_gl = abs_gains_and_losses[parameter][param_value].sel(agent=['ALL_INDUSTRY:{}'.format(r) for r in region_group])
                abs_gains = abs_gl.values[abs_gl.values > 0].sum() / 1e6
                abs_losses = abs_gl.values[abs_gl.values < 0].sum() / 1e6
                ax = axs2[row_idx, col_idx]
                ax.bar(param_value, abs_gains, color='orange', width=col_width[parameter], alpha=.5)
                ax.bar(param_value, abs_losses, color='purple', width=col_width[parameter], alpha=.5)
                ax.bar(param_value, abs_gains + abs_losses, color='k', alpha=0.3, width=col_width[parameter])
                ax.set_xlabel(parameter_letters[parameter], fontdict={'fontsize': FSIZE_MEDIUM})
                ax.axvline(default_params[parameter], linestyle='dotted', color='k', lw=1)
                ax.axhline(0, color='k', lw=0.75)
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(tick_formatters[parameter][0]))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_formatters[parameter][1]))
                fig2_data_output.loc[(parameter, param_value, region_group_name), :] = [
                    abs_gains,
                    abs_losses,
                    abs_gains + abs_losses
                ]
    title_transform = transforms.blended_transform_factory(axs2[0, 0].transAxes, fig2.transFigure)
    fig2.text(0.5, 1, 'Global', fontdict={'fontsize': FSIZE_MEDIUM}, transform=title_transform, ha='center', va='top')
    title_transform = transforms.blended_transform_factory(axs2[0, 1].transAxes, fig2.transFigure)
    fig2.text(0.5, 1, 'USA', fontdict={'fontsize': FSIZE_MEDIUM}, transform=title_transform, ha='center', va='top')
    for ax_idx, ax in enumerate(axs2.flatten()):
        ax.text(0, 1, chr(ax_idx + 97), fontweight='bold', ha='left', va='bottom', transform=ax.transAxes)
    axs2[int(len(axs1) / 2), 0].set_ylabel('gains / losses [bn USD]', fontdict={'fontsize': FSIZE_MEDIUM})
    fig2.tight_layout()
    for col, factor in enumerate(np.arange(len(axs2), 0, -1) - 1):
        for ax in axs2[col]:
            x, y, w, h = ax.get_position().x0, ax.get_position().y0, ax.get_position().width, ax.get_position().height
            shrink = -0.01
            move = 0.01
            ax.set_position([x, y - move * factor, w, h - shrink])

    fig3_data_output = pd.Series(
        index=pd.MultiIndex(levels=[[], [], []], codes=[[], [], []], names=[u'parameter', u'param_value', u'region_group']),
        name='value',
        dtype='float64'
    )
    fig3_size = (MAX_FIG_WIDTH_WIDE, MAX_FIG_WIDTH_WIDE / 3.25)
    fig3_aspect = fig3_size[0] / fig3_size[1]
    fig3, axs3 = plt.subplots(nrows=1, ncols=len(data.keys()) + 1, sharex=False, sharey=False,
                              figsize=fig3_size, gridspec_kw={'width_ratios': [1, 1, 1, 0.25]})
    region_groups = ['USA', 'AI:50', 'AI:75', 'AI:95']
    for ax_idx, parameter in tqdm.tqdm(enumerate(data.keys())):
        for param_value, dataset in sorted(list(data[parameter].items())):
            abs_gl = abs_gains_and_losses[parameter][param_value]
            total_gains = abs_gl.sel(agent=list(set(abs_gl.agent.values) - {'ALL_INDUSTRY:US.TX', 'ALL_INDUSTRY:US.LA'})).sum().values.item()
            previous_share = 0
            for rg_index, region_group in enumerate(region_groups):
                agents = ['ALL_INDUSTRY:{}'.format(r) for r in np.intersect1d(WORLD_REGIONS[region_group], abs_gl.agent_region.values)]
                agents = list(set(agents) - {'ALL_INDUSTRY:US.TX', 'ALL_INDUSTRY:US.LA'})
                group_gains = abs_gl.sel(agent=agents).sum().values.item()
                gain_share = group_gains / total_gains * 100 - previous_share
                axs3[ax_idx].bar(x=param_value, height=gain_share, width=col_width[parameter], bottom=previous_share,
                                 align='center', color=paramter_colors[ax_idx], alpha=(1 - rg_index * 0.3))
                previous_share += gain_share
                fig3_data_output.loc[(parameter, param_value, region_group)] = gain_share
        axs3[ax_idx].set_xlabel(parameter_letters[parameter], fontdict={'fontsize': FSIZE_MEDIUM})
        axs3[ax_idx].axvline(default_params[parameter], linestyle='dotted', color='k', lw=1)
    axs3[0].set_ylabel('gain share [%]')
    plt.tight_layout()
    l_ax = axs3[-1]
    l_ax.axis('off')
    l_ax_x0 = axs3[-2].get_position().x1
    l_ax_width = 1 - l_ax_x0
    l_ax_height = l_ax_width * fig3_aspect
    l_ax.set_position([l_ax_x0, axs3[-2].get_position().y1 - l_ax_height, l_ax_width, l_ax_height])
    r_width, r_height = 0.1, 0.1
    for param_idx, parameter in enumerate(data.keys()):
        for rg_idx, region_group in enumerate(region_groups):
            x = 0.2 + param_idx * r_width
            y = 1 - (len(region_groups) - rg_idx) * r_height
            rect = patches.Rectangle((x, y), r_width, r_height, edgecolor=None, linewidth=0,
                                     facecolor=paramter_colors[param_idx], alpha=(1 - rg_idx * 0.3),
                                     transform=l_ax.transAxes)
            l_ax.add_patch(rect)
            if param_idx == len(data.keys()) - 1:
                l_ax.text(x + 0.11, y + r_height / 2, region_group, ha='left', va='center', transform=l_ax.transAxes)
    for ax_idx, ax in enumerate(axs3.flatten()[:3]):
        ax.text(0, 1, chr(ax_idx + 97), fontweight='bold', ha='left', va='bottom', transform=ax.transAxes)

    if name is not None:
        fig1.savefig("/home/robin/repos/harvey_scaling/figures/figures/sensitivity_analysis/time_series_{}.pdf".format(name),
                     dpi=300)
        fig2.savefig("/home/robin/repos/harvey_scaling/figures/figures/sensitivity_analysis/gains_losses_{}.pdf".format(name),
                     dpi=300)
        fig3.savefig("/home/robin/repos/harvey_scaling/figures/figures/sensitivity_analysis/gain_shares_{}.pdf".format(name),
                     dpi=300)
    plt.show()
    if store_data is not None:
        for fig_number, output_data in zip(fig_numbers, [fig1_data_output, fig2_data_output, fig3_data_output]):
            output_data.to_csv(store_data.format(fig_number))
    return data


def load_simulation_data():
    full_data_ensemble_path = "/home/robin/repos/harvey_scaling/data/acclimate_output/main_analysis/HARVEY_econYear2015_dT_0_3.2_0.2__re0_40000.0_2500.0__old_acclimate__ccFactor1.19"
    slope_ensemble_path = "/home/robin/repos/harvey_scaling/data/acclimate_output/main_analysis/HARVEY_econYear2015_dT_0_3.2_0.8__slopes0+6250+12500__old_acclimate__ccFactor1.19/"
    datasets = []
    for ensemble_path in [full_data_ensemble_path, slope_ensemble_path]:
        datasets.append(pickle.load(open(ensemble_path + "/production_consumption_netProduction__ALL_INDUSTRY_MINQ.pk", 'rb')))
        meta = pickle.load(open(ensemble_path + "/ensemble_meta.pk", 'rb'))
        if 'slopes' in meta.keys():
            datasets[-1].slope_meta = meta['slopes']
        datasets[-1].scaled_scenarios = meta['scaled_scenarios']
        datasets[-1].calc_sector_diff('ALL_INDUSTRY', 'MINQ', _inplace=True)
        clean_regions(datasets[-1])
    return datasets[0], datasets[1]


if __name__ == '__main__':
    data, slope_data = load_simulation_data()

    fig1a_data = make_agent_var_global_map(data, _sector='ALL_INDUSTRY-MINQ', _variable='firms_production-firms_consumption', _numbering='a', _cbar_lims=[-100, 12], _symmetric_cbar=True, _outfile="../figures/figures/maps/global_map_production_change_PRIVSECTORS-MINQ_dt0_re0.pdf")
    fig1b_data = make_agent_var_global_map(data, _sector='MINQ', _variable='firms_production-firms_consumption', _numbering='b', _cbar_lims=[-100, 12], _symmetric_cbar=True, _outfile="../figures/figures/maps/global_map_production_change_MINQ_dt0_re0.pdf")
    fig1a_data.to_csv("/home/robin/repos/harvey_scaling/figures/figure_data/1a.csv")
    fig1b_data.to_csv("/home/robin/repos/harvey_scaling/figures/figure_data/1b.csv")

    plot_initial_claims(_outfile="/home/robin/repos/harvey_scaling/figures/figures/harvey_initial_claims.pdf")

    plot_radius_extension_map(_numbering='a', _outfile="/home/robin/repos/harvey_scaling/figures/figures/radius_map.pdf")
    plot_radius_extension_impact(_numbering='b', _outfile="/home/robin/repos/harvey_scaling/figures/figures/radius_extension_impact.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/supp6b.csv")

    sst_gmt_relationship(sea='Gulf of Mexico', hurricane_months_only=True, regression=True, monthly_diff_to_historical=True, outfile="/home/robin/repos/harvey_scaling/figures/figures/gulf_of_mex_sst_gmt_relationship.pdf")

    make_heatmap(data.get_regions('WORLD_wo_TX_LA').get_vars('firms_production-firms_consumption').get_sectors('ALL_INDUSTRY-MINQ').clip(4, 365 + 4).aggregate('relative_difference'), _xlabel=None, _ylabel='Global relative production\ndifference (% baseline)', _numbering='a', _slope_data=slope_data.get_regions('WORLD_wo_TX_LA').get_vars('firms_production-firms_consumption').get_sectors('ALL_INDUSTRY-MINQ').clip(4, 365+4).aggregate('relative_difference'), _gauss_filter=False, _sst_gmt_factor=0.8, _y_ax_precision=3, _outfile="/home/robin/repos/harvey_scaling/figures/figures/heatmaps/heatmap_PRIVSECTORS-MINQ_production_relative_difference_ROW_w_USA.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/2a.csv")
    make_heatmap_cut(data.get_regions('WORLD_wo_TX_LA').get_vars('firms_production-firms_consumption').get_sectors('ALL_INDUSTRY-MINQ').clip(4, 365 + 4).aggregate('relative_difference'), _xlabel=r'$\Delta T$ (temperature change in °C)', _ylabel='Global relative production\ndifference (% baseline)', _numbering='c', _slope_data=slope_data.get_regions('WORLD_wo_TX_LA').get_vars('firms_production-firms_consumption').get_sectors('ALL_INDUSTRY-MINQ').clip(4, 365+4).aggregate('relative_difference'), _gauss_filter=False, _sst_gmt_factor=0.8, _y_ax_precision=3, _outfile="/home/robin/repos/harvey_scaling/figures/figures/heatmaps/heatmap_cut_PRIVSECTORS-MINQ_production_relative_difference_ROW_w_USA.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/2c.csv")
    make_heatmap(data.get_regions('WORLD_wo_TX_LA').get_vars('firms_production-firms_consumption').get_sectors('MINQ').clip(4, 365 + 4).aggregate('relative_difference'), _xlabel=None, _ylabel='Global relative production\ndifference (% baseline)', _numbering='b', _slope_data=slope_data.get_regions('WORLD_wo_TX_LA').get_vars('firms_production-firms_consumption').get_sectors('MINQ').clip(4, 365+4).aggregate('relative_difference'), _gauss_filter=False, _sst_gmt_factor=0.8, _y_ax_precision=3, _outfile="/home/robin/repos/harvey_scaling/figures/figures/heatmaps/heatmap_MINQ_production_relative_difference_ROW_w_USA.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/2b.csv")
    make_heatmap_cut(data.get_regions('WORLD_wo_TX_LA').get_vars('firms_production-firms_consumption').get_sectors('MINQ').clip(4, 365 + 4).aggregate('relative_difference'), _xlabel=r'$\Delta T$ (temperature change in °C)', _ylabel='Global relative production\ndifference (% baseline)', _numbering='d', _slope_data=slope_data.get_regions('WORLD_wo_TX_LA').get_vars('firms_production-firms_consumption').get_sectors('MINQ').clip(4, 365+4).aggregate('relative_difference'), _gauss_filter=False, _sst_gmt_factor=0.8, _y_ax_precision=3, _legend=False, _outfile="/home/robin/repos/harvey_scaling/figures/figures/heatmaps/heatmap_cut_MINQ_production_relative_difference_ROW_w_USA.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/2d.csv")

    plot_gains_and_losses_from_slope_dataset(slope_data.get_vars('firms_production-firms_consumption').get_sectors('ALL_INDUSTRY-MINQ'), 'WORLD', data.get_vars('firms_production-firms_consumption').get_sectors('ALL_INDUSTRY-MINQ'), _sst_gmt_factor=0.8, _numbering='a', _sector_label=True, _xlabel=False, _ylabel=True, _ax_right=False, _upper_slope_labels=True, _outfile="/home/robin/repos/harvey_scaling/figures/figures/region_gains_and_losses/gains_losses_WORLD_PRIVSECTORS-MINQ.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/3a.csv")
    plot_gains_and_losses_from_slope_dataset(slope_data.get_vars('firms_production-firms_consumption').get_sectors('MINQ'), 'WORLD', data.get_vars('firms_production-firms_consumption').get_sectors('MINQ'), _sst_gmt_factor=0.8, _numbering='b', _sector_label=True, _xlabel=False, _ylabel=False, _ax_right=True, _upper_slope_labels=True, _major_ytick_loc=1, _outfile="/home/robin/repos/harvey_scaling/figures/figures/region_gains_and_losses/gains_losses_WORLD_MINQ.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/3b.csv")
    plot_gains_and_losses_from_slope_dataset(slope_data.get_vars('firms_production-firms_consumption').get_sectors('ALL_INDUSTRY-MINQ'), 'USA', data.get_vars('firms_production-firms_consumption').get_sectors('ALL_INDUSTRY-MINQ'), _sst_gmt_factor=0.8, _numbering='c', _sector_label=False, _xlabel=True, _ylabel=True, _ax_right=False, _upper_slope_labels=False, _outfile="/home/robin/repos/harvey_scaling/figures/figures/region_gains_and_losses/gains_losses_USA_PRIVSECTORS-MINQ.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/3c.csv")
    plot_gains_and_losses_from_slope_dataset(slope_data.get_vars('firms_production-firms_consumption').get_sectors('MINQ'), 'USA', data.get_vars('firms_production-firms_consumption').get_sectors('MINQ'), _sst_gmt_factor=0.8, _numbering='d', _sector_label=False, _xlabel=True, _ylabel=False, _ax_right=True, _upper_slope_labels=False, _minor_ytick_loc=0.5, _major_ytick_loc=1, _outfile="/home/robin/repos/harvey_scaling/figures/figures/region_gains_and_losses/gains_losses_USA_MINQ.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/3d.csv")

    plot_global_shares_from_slope_dataset(slope_data.get_vars('firms_production-firms_consumption').get_sectors('ALL_INDUSTRY-MINQ'), _region_groups=['USA', 'AI-MQ:50', 'AI-MQ:75', 'AI-MQ:95'], _sst_gmt_factor=0.8, _bar_reg_label=False, _numbering='a', _relative=True, _xlabel=False, _ylabel = True, _inplot_legend = False, _ylim=(50, 99), _minor_ytick_loc=None, _major_ytick_loc=10, _ax_right=False, _plot_regression={'AI-MQ:50': ['A', 'B', 'C', 'D'], 'AI-MQ:75': ['A', 'B','C', 'D']}, _outfile="/home/robin/repos/harvey_scaling/figures/figures/gain_shares/gain_shares_PRIVSECTORS-MINQ_AI-MQregions_rel.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/4a.csv")
    plot_global_shares_from_slope_dataset(slope_data.get_vars('firms_production-firms_consumption').get_sectors('ALL_INDUSTRY-MINQ'), _region_groups=['USA', 'AI-MQ:50', 'AI-MQ:75', 'AI-MQ:95'], _sst_gmt_factor=0.8, _bar_reg_label=False, _numbering='c', _relative=False, _xlabel=True, _ylabel = True, _inplot_legend = True, _ylim=(10, 35), _minor_ytick_loc=None, _major_ytick_loc=10, _inplot_sector_label=False, _ax_right=False, _outfile="/home/robin/repos/harvey_scaling/figures/figures/gain_shares/gain_shares_PRIVSECTORS-MINQ_AI-MQregions_abs.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/4c.csv")
    plot_global_shares_from_slope_dataset(slope_data.get_vars('firms_production-firms_consumption').get_sectors('MINQ'), _region_groups=['MQ:50', 'MQ:50+USA', 'MQ:75', 'MQ:95'], _sst_gmt_factor=0.8, _bar_reg_label=False, _numbering='b', _relative=True, _xlabel=False, _ylabel = False, _inplot_legend = False, _ylim=(20, 100), _minor_ytick_loc=None, _major_ytick_loc=10, _ax_right=True, _plot_regression={'MQ:50+USA': ['A', 'B', 'C', 'D'], 'MQ:75': ['A', 'B','C', 'D']}, _outfile="/home/robin/repos/harvey_scaling/figures/figures/gain_shares/gain_shares_MINQ_MQregions_rel.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/4b.csv")
    plot_global_shares_from_slope_dataset(slope_data.get_vars('firms_production-firms_consumption').get_sectors('MINQ'), _region_groups=['MQ:50', 'MQ:50+USA', 'MQ:75', 'MQ:95'], _sst_gmt_factor=0.8, _bar_reg_label=False, _numbering='d', _relative=False, _xlabel=True, _ylabel = False, _inplot_legend = True, _ylim=(0, 2.1), _minor_ytick_loc=None, _major_ytick_loc=0.5, _inplot_sector_label=False, _ax_right=True, _outfile="/home/robin/repos/harvey_scaling/figures/figures/gain_shares/gain_shares_MINQ_MQregions_abs.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/4d.csv")

    plot_global_shares_from_slope_dataset(slope_data.get_vars('firms_production-firms_consumption').get_sectors('ALL_INDUSTRY-MINQ'), _region_groups=['USA', 'AI-MQ:50', 'AI-MQ:75', 'AI-MQ:95'], _sst_gmt_factor=0.8, _bar_reg_label=False, _inplot_legend=False, _numbering='a', _relative = True, _xlabel = False, _ylim=(50, 99), _major_ytick_loc=10, _outfile="/home/robin/repos/harvey_scaling/figures/figures/gain_shares/gain_shares_supp_PRIVSECTORS-MINQ_AI-MQregions_rel.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/supp1a.csv")
    plot_global_shares_from_slope_dataset(slope_data.get_vars('firms_production-firms_consumption').get_sectors('MINQ'), _region_groups=['USA', 'AI-MQ:50', 'AI-MQ:75', 'AI-MQ:95'], _sst_gmt_factor=0.8, _bar_reg_label=False, _inplot_legend=False, _numbering = 'b', _ylabel = False, _xlabel = False, _ylim=(0, 80), _ax_right=True, _outfile="/home/robin/repos/harvey_scaling/figures/figures/gain_shares/gain_shares_supp_MINQ_AI-MQregions_rel.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/supp1b.csv")
    plot_global_shares_from_slope_dataset(slope_data.get_vars('firms_production-firms_consumption').get_sectors('ALL_INDUSTRY-MINQ'), _region_groups=['USA', 'AI-MQ:50', 'AI-MQ:75', 'AI-MQ:95'], _sst_gmt_factor=0.8, _bar_reg_label=False, _numbering='c', _relative = False, _ylabel = True, _ylim=(10, 35), _minor_ytick_loc=None, _major_ytick_loc=10, _inplot_legend = True, _inplot_sector_label=False, _outfile="/home/robin/repos/harvey_scaling/figures/figures/gain_shares/gain_shares_supp_PRIVSECTORS-MINQ_AI-MQregions_abs.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/supp1c.csv")
    plot_global_shares_from_slope_dataset(slope_data.get_vars('firms_production-firms_consumption').get_sectors('MINQ'), _region_groups=['USA', 'AI-MQ:50', 'AI-MQ:75', 'AI-MQ:95'], _sst_gmt_factor=0.8, _bar_reg_label=False, _numbering='d', _relative=False, _ylabel = False, _ylim=None, _minor_ytick_loc=None, _major_ytick_loc=0.5, _ax_right=True, _inplot_legend = False, _inplot_sector_label=False, _outfile="/home/robin/repos/harvey_scaling/figures/figures/gain_shares/gain_shares_supp_MINQ_AI-MQregions_abs.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/supp1d.csv")

    plot_global_shares_from_slope_dataset(slope_data.get_vars('firms_production-firms_consumption').get_sectors('ALL_INDUSTRY-MINQ'), _region_groups=['MQ:50', 'MQ:50+USA', 'MQ:75', 'MQ:95'], _sst_gmt_factor=0.8, _bar_reg_label=False, _inplot_legend=False, _numbering='a', _relative = True, _xlabel = False, _ylim=(0, 99), _major_ytick_loc=10, _outfile="/home/robin/repos/harvey_scaling/figures/figures/gain_shares/gain_shares_supp_PRIVSECTORS-MINQ_MQregions_rel.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/supp2a.csv")
    plot_global_shares_from_slope_dataset(slope_data.get_vars('firms_production-firms_consumption').get_sectors('MINQ'), _region_groups=['MQ:50', 'MQ:50+USA', 'MQ:75', 'MQ:95'], _sst_gmt_factor=0.8, _bar_reg_label=False, _inplot_legend=False, _numbering = 'b', _ylabel = False, _xlabel = False, _ylim=(20, 100), _ax_right=True, _outfile="/home/robin/repos/harvey_scaling/figures/figures/gain_shares/gain_shares_supp_MINQ_MQregions_rel.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/supp2b.csv")
    plot_global_shares_from_slope_dataset(slope_data.get_vars('firms_production-firms_consumption').get_sectors('ALL_INDUSTRY-MINQ'), _region_groups=['MQ:50', 'MQ:50+USA', 'MQ:75', 'MQ:95'], _sst_gmt_factor=0.8, _bar_reg_label=False, _numbering='c', _relative = False, _ylabel = True, _ylim=(0, 35), _minor_ytick_loc=None, _major_ytick_loc=10, _inplot_legend = True, _inplot_sector_label=False, _outfile="/home/robin/repos/harvey_scaling/figures/figures/gain_shares/gain_shares_supp_PRIVSECTORS-MINQ_MQregions_abs.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/supp2c.csv")
    plot_global_shares_from_slope_dataset(slope_data.get_vars('firms_production-firms_consumption').get_sectors('MINQ'), _region_groups=['MQ:50', 'MQ:50+USA', 'MQ:75', 'MQ:95'], _sst_gmt_factor=0.8, _bar_reg_label=False, _numbering='d', _relative=False, _ylabel = False, _ylim=(0, 1.8), _minor_ytick_loc=None, _major_ytick_loc=0.5, _ax_right=True, _inplot_legend = False, _inplot_sector_label=False, _outfile="/home/robin/repos/harvey_scaling/figures/figures/gain_shares/gain_shares_supp_MINQ_MQregions_abs.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/supp2d.csv")

    plot_compensation_gap(slope_data.get_vars('firms_production-firms_consumption').get_sectors('ALL_INDUSTRY-MINQ'), region='USA', _sst_gmt_factor=0.8, _numbering='a', _sector_label=True, _xlabel=False, _legend=True, _outfile="/home/robin/repos/harvey_scaling/figures/figures/compensation_gap/compensation_gap_PRIVSECTORS-MINQ_USA.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/5a.csv")
    plot_compensation_gap(slope_data.get_vars('firms_production-firms_consumption').get_sectors('MINQ'), region='USA', _sst_gmt_factor=0.8, _numbering='b', _sector_label=True, _xlabel=False, _ylabel=False, _legend=False, _ax_right=True, _outfile="/home/robin/repos/harvey_scaling/figures/figures/compensation_gap/compensation_gap_MINQ_USA.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/5b.csv")
    plot_compensation_gap(slope_data.get_vars('firms_production-firms_consumption').get_sectors('ALL_INDUSTRY-MINQ'), region='WORLD', _sst_gmt_factor=0.8, _numbering='c', _sector_label=False, _xlabel=True, _legend=False, _remove_upper_tick=True, _outfile="/home/robin/repos/harvey_scaling/figures/figures/compensation_gap/compensation_gap_PRIVSECTORS-MINQ_WORLD.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/5c.csv")
    plot_compensation_gap(slope_data.get_vars('firms_production-firms_consumption').get_sectors('MINQ'), region='WORLD', _sst_gmt_factor=0.8, _numbering='d', _sector_label=False, _xlabel=True, _legend=False, _ylabel=False, _ax_right=True, _outfile="/home/robin/repos/harvey_scaling/figures/figures/compensation_gap/compensation_gap_MINQ_WORLD.pdf", store_data="/home/robin/repos/harvey_scaling/figures/figure_data/5d.csv")

    plot_sensitivity_analysis(path="/home/robin/repos/harvey_scaling/data/acclimate_output/sensitivity_analysis/2022-05-13_18:58:04unscaled/", name='unscaled', store_data="/home/robin/repos/harvey_scaling/figures/figure_data/supp{}.csv", fig_numbers=[7, 9, 11])
    plot_sensitivity_analysis(path="/home/robin/repos/harvey_scaling/data/acclimate_output/sensitivity_analysis/2022-05-13_18:44:11maximum_scaled/", name='max_scaled', store_data="/home/robin/repos/harvey_scaling/figures/figure_data/supp{}.csv", fig_numbers=[8, 10, 12])
    pass

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import os
import sys
sys.path.append("../")
from data.calc_initial_forcing_intensity_HARVEY import plot_polygon, load_hwm, alpha_shape, alpha
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


if __name__ == '__main__':
    # plot_initial_claims(_outfile="../figures/harvey_initial_claims.pdf")
    pass
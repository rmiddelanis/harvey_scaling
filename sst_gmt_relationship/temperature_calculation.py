import matplotlib
import tqdm
from netCDF4 import Dataset
import geopandas as gpd
import itertools
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

from scipy import stats

sys.path.append("{}/repos/harvey_scaling/sst_gmt_relationship".format(os.path.expanduser('~')))
from cmip_aggregation import aggregated_files_path
from grid_weights import calc_grid_sizes, global_land_shares_path, sea_land_shares_path
import pandas as pd
import seaborn as sns

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

time_aggr = 'monthly'
lat_nor = 23.5
lat_sou = -23.5


def calc_mean_temperatures(_datasets: [Dataset], _sea_selection=None, _lat_min=None, _lat_max=None):
    global_weight_gdf = gpd.read_file(global_land_shares_path)
    if 'area' not in global_weight_gdf.columns:
        calc_grid_sizes(global_weight_gdf)

    if _sea_selection is not None:
        sea_weight_lonlats = gpd.read_file(sea_land_shares_path.format(_sea_selection))
    else:
        sea_weight_lonlats = gpd.read_file(global_land_shares_path)
    sea_weight_lonlats = sea_weight_lonlats[sea_weight_lonlats['land_share'] != 1]
    if _lat_min is not None:
        sea_weight_lonlats = sea_weight_lonlats[sea_weight_lonlats['lat'] >= _lat_min]
    if _lat_max is not None:
        sea_weight_lonlats = sea_weight_lonlats[sea_weight_lonlats['lat'] <= _lat_max]
    sea_weight_lonlats = sea_weight_lonlats.apply(lambda x: (x['lon'], x['lat']), axis=1)

    sea_weight_gdf_selection = global_weight_gdf['land_share'] != 1
    sea_weight_gdf_selection = sea_weight_gdf_selection & global_weight_gdf.apply(lambda x: (x['lon'], x['lat']),
                                                                                  axis=1).isin(sea_weight_lonlats)
    global_weight_gdf['weights_sea'] = (- global_weight_gdf['land_share'] + 1) * global_weight_gdf[
        'area'] * sea_weight_gdf_selection
    global_weight_gdf['weights_sea'] = global_weight_gdf['weights_sea'] / global_weight_gdf['weights_sea'].sum()

    land_weight_gdf_selection = global_weight_gdf['land_share'] != 0
    global_weight_gdf['weights_land'] = global_weight_gdf['land_share'] * global_weight_gdf[
        'area'] * land_weight_gdf_selection
    global_weight_gdf['weights_land'] = global_weight_gdf['weights_land'] / global_weight_gdf['weights_land'].sum()

    global_weight_gdf['weights_global'] = global_weight_gdf['area']
    global_weight_gdf['weights_global'] = global_weight_gdf['weights_global'] / global_weight_gdf[
        'weights_global'].sum()

    global_weight_gdf['lonlat'] = global_weight_gdf.apply(lambda x: (x['lon'], x['lat']), axis=1)
    # global_weight_gdf.set_index('lonlat')

    all_lonlats = list(itertools.product(_datasets[0]['lon'][:], _datasets[0]['lat'][:]))
    lonlat_df = pd.DataFrame()
    lonlat_df['lonlat'] = all_lonlats
    lonlat_df['temperature_index'] = np.arange(len(all_lonlats))
    lonlat_df.set_index('lonlat', inplace=True)
    global_weight_gdf = global_weight_gdf.join(lonlat_df, on='lonlat')
    data_slice = global_weight_gdf['temperature_index']

    res = []
    for _data in tqdm.tqdm(_datasets):
        temperature = _data['tas'][:].transpose((0, 2, 1))
        temperature = temperature.reshape(temperature.shape[0], -1)
        mean_gmt = (temperature[:, data_slice].data * global_weight_gdf['weights_global'].values).sum(axis=-1)
        mean_sst = (temperature[:, data_slice].data * global_weight_gdf['weights_sea'].values).sum(axis=-1)
        mean_lst = (temperature[:, data_slice].data * global_weight_gdf['weights_land'].values).sum(axis=-1)
        res.append((mean_gmt, mean_sst, mean_lst))
    return res


def make_sst_gmt_scatter(ssp_list=None, regression=True, xlabel=None, ylabel=None, month_from=1, month_to=12,
                         model_selection=None, _color_months=True, **kwargs):
    if ssp_list is None:
        ssp_list = ['ssp126', 'ssp370', 'ssp585']
    if model_selection is None:
        model_selection = ['GFDL-ESM4', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'UKESM1-0-LL']

    datasets = {}
    for ssp in ssp_list:
        datasets[ssp] = {}
        ssp_dir = os.path.join(aggregated_files_path, time_aggr, ssp)
        for model_name in os.listdir(ssp_dir):
            model_dir = os.path.join(ssp_dir, model_name)
            if os.path.isdir(model_dir) and model_name in model_selection:
                datasets[ssp][model_name] = []
                for filename in os.listdir(model_dir):
                    if '_tas_' in filename:
                        datasets[ssp][model_name].append(Dataset(os.path.join(ssp_dir, model_name, filename)))

    fig_width = MAX_FIG_WIDTH_WIDE
    fig_height = MAX_FIG_WIDTH_WIDE * 0.5
    fig, axs = plt.subplots(ncols=len(ssp_list), sharey=True, sharex=True, figsize=(fig_width, fig_height),
                            constrained_layout=True)
    if len(ssp_list) == 1:
        axs = [axs]

    month_cmap = plt.cm.get_cmap('viridis')
    month_colors = [month_cmap(i / 12) for i in range(12)]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_mask = ([False] * (month_from - 1) + [True] * (month_to - (month_from - 1)) + [False] * (12 - month_to))

    all_temperatures = []
    markers = ['d', '^', 'o', 's', 'p']
    for ssp_idx, ssp in enumerate(ssp_list):
        ssp_datasets = []
        for model_name in datasets[ssp].keys():
            ssp_datasets += datasets[ssp][model_name]
        temperatures = calc_mean_temperatures(ssp_datasets, **kwargs)
        all_temperatures.append(temperatures)
        all_xvals = []
        all_yvals = []
        for model_idx, model_name in enumerate(datasets[ssp].keys()):
            num_datasets = len(datasets[ssp][model_name])
            model_results = temperatures[:num_datasets]
            if len(temperatures) > num_datasets:
                temperatures = temperatures[num_datasets:]
            for model_result_idx, model_result in enumerate(model_results):
                month_selection = month_mask * int(len(model_result[0]) / 12)
                gmt = model_result[0][month_selection]
                sst = model_result[1][month_selection]
                plot_colors = np.array(month_colors * int(len(model_result[0]) / 12))[month_selection]
                for idx in range(len(gmt)):
                    if model_result_idx == 0 and idx == 0:
                        label = model_name
                    else:
                        label = None
                    if _color_months:
                        color = plot_colors[idx]
                    else:
                        color = 'k'
                    axs[ssp_idx].scatter(gmt[idx], sst[idx], marker=markers[model_idx], label=label, facecolors='none',
                                         edgecolors=color, s=10, linewidths=0.5, alpha=0.2)
                # if model_result_idx == 0:
                #     label = model_name
                # else:
                #     label = None
                # axs[ssp_idx].scatter(gmt, sst, marker=markers[model_idx], label=label, facecolors='none',
                #                      edgecolors='k', s=10, linewidths=0.5, alpha=0.2)

                all_xvals += list(gmt)
                all_yvals += list(sst)
        if regression:
            # sns.regplot(all_xvals, all_yvals, scatter=False, ax=axs[ssp_idx], ci=99)
            reg = stats.linregress(all_xvals, all_yvals)
            axs[ssp_idx].plot(np.array(all_xvals), reg.intercept + reg.slope * np.array(all_xvals), 'r')
            stats_legend = "y = {0:.3f}x + {1:.3f}\nr = {2:.3f}".format(reg.slope, reg.intercept, reg.rvalue)
            axs[ssp_idx].text(0.99, 0.01, stats_legend, ha='right', va='bottom', transform=axs[ssp_idx].transAxes,
                              fontsize=FSIZE_TINY, color='r')
    legend = axs[0].legend()
    for lh in legend.legendHandles:
        lh.set_alpha(1)
    if xlabel is not None:
        axs[0].set_ylabel(ylabel)
    if ylabel is not None:
        axs[int(np.floor(len(axs) / 2))].set_xlabel(xlabel)

    if _color_months:
        fig.colorbar()

    plt.tight_layout()

    for ax_idx, ax in enumerate(axs):
        pos_old = ax.get_position()
        height_new = pos_old.height * 0.99
        ax.set_position([pos_old.x0, pos_old.y0, pos_old.width, height_new])
        fig.text(pos_old.x0, 1, chr(ax_idx + 97), ha='center', va='top', fontweight='bold')

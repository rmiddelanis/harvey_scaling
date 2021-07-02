import argparse
from datetime import datetime, timedelta
import tqdm
from dateutil.relativedelta import relativedelta
from netCDF4 import Dataset, date2num
import os
import calendar
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from cartopy import config
import cartopy.crs as ccrs

isimip_path = "/p/projects/isimip/isimip/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily"
aggregated_files_path = "{}/repos/harvey_scaling/sst_gmt_relationship/data/isimip/aggregated/".format(
    os.path.expanduser('~'))
ssp_list = 'ssp126+ssp370+ssp585'
var_list = 'tas'
time_aggr = 'monthly'


def get_path_list(_isimip_path, _ssp, _var):
    ssp_path = os.path.join(_isimip_path, _ssp)
    res = {}
    for dir_entry in os.listdir(ssp_path):
        entry_path = os.path.join(ssp_path, dir_entry)
        if os.path.isdir(entry_path):
            _model_path = entry_path
            model_files = []
            for file in os.listdir(_model_path):
                file_path = os.path.join(_model_path, file)
                if os.path.isfile(file_path) and file.find('_' + _var + '_') != -1:
                    model_files.append(file_path)
            res[dir_entry] = model_files
    return res


def aggregate_dataset(_dataset, _start_year, _end_year, _outfile, _varname, _time_aggr='monthly'):
    day_intervals = np.array(
        [calendar.monthrange(y, m)[1] for m in range(1, 13) for y in range(_start_year, _end_year + 1)])
    if _time_aggr == 'yearly':
        day_intervals = day_intervals.reshape(12, -1).sum(axis=0)
        date_vals = [datetime(_start_year, 1, 1) + n * relativedelta(years=1) for n in
                     range((_end_year - _start_year + 1))]
    elif _time_aggr == 'monthly':
        date_vals = [datetime(_start_year, 1, 1) + n * relativedelta(months=1) for n in
                     range((_end_year - _start_year + 1) * 12)]

    d_agg = Dataset(_outfile, 'w', format='NETCDF4')

    d_agg.createDimension('lon', _dataset.dimensions['lon'].size)
    d_agg.createDimension('lat', _dataset.dimensions['lat'].size)
    d_agg.createDimension('time', None)

    lons = d_agg.createVariable('lon', 'f8', ('lon',))
    lons[:] = _dataset['lon'][:]
    lats = d_agg.createVariable('lat', 'f8', ('lat',))
    lats[:] = _dataset['lat'][:]

    times = d_agg.createVariable('time', 'f8', ('time',))
    times.calendar = 'proleptic_gregorian'
    times.units = 'months  as  %Y%m'
    times[:] = [int(date_val.year * 100 + date_val.month) for date_val in date_vals]
    # times[:] = np.arange(len(day_intervals))

    variable = d_agg.createVariable(_varname, 'f4', ('time', 'lat', 'lon'))
    variable.units = _dataset[_varname].units
    variable.standard_name = _dataset[_varname].standard_name
    variable.long_name = _dataset[_varname].long_name

    for interval_idx, interval_duration in enumerate(day_intervals):
        day_from = day_intervals[:interval_idx].sum()
        day_to = day_from + interval_duration
        variable[interval_idx, :, :] = _dataset[_varname][day_from:day_to, :, :].mean(axis=0)
    return d_agg


def aggregate_file(_input_path, _model_output_dir, _varname):
    year_from = int(_input_path[-12:-8])
    year_to = int(_input_path[-7:-3])
    outfile = os.path.join(_model_output_dir, _input_path.split('/')[-1].replace('daily', time_aggr))
    dataset = Dataset(_input_path)
    d_agg = aggregate_dataset(dataset, _start_year=year_from, _end_year=year_to, _outfile=outfile, _varname=_varname,
                              _time_aggr=time_aggr)
    d_agg.close()
    dataset.close()
    print('wrote {}'.format(outfile))


def parallel_aggregation(_vars, _ssps):
    starmap_parameters = []
    for var in _vars:
        for ssp in _ssps:
            input_path_dict = get_path_list(isimip_path, ssp, var)
            for model, input_paths in input_path_dict.items():
                model_output_dir = os.path.join(aggregated_files_path, time_aggr, ssp, model)
                if not os.path.exists(model_output_dir):
                    os.makedirs(model_output_dir)
                for input_path in input_paths:
                    starmap_parameters.append((input_path, model_output_dir, var))
    with Pool() as pool:
        pool.starmap(aggregate_file, starmap_parameters)


def plot_var(_dataset, _t, _var):
    fig = plt.figure(figsize=(8, 4))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(resolution="110m", linewidth=1)
    ax.gridlines(linestyle='--', color='black')
    lons = _dataset['lon'][:]
    lats = _dataset['lat'][:]
    data = _dataset[_var][_t, ...]
    plt.contourf(lons, lats, data, transform=ccrs.PlateCarree(), cmap=plt.cm.jet)
    cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    date = str(int(_dataset['time'][_t].item()))
    year = date[:4]
    month = ''
    day = ''
    if len(date) > 4:
        month = '-' + date[4:6]
        if len(date) > 6:
            month = '-' + date[6:]
    plt.title('Temperature on {}{}{}'.format(year, month, day), size=14)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--isimip_path', type=str, default=isimip_path, help='')
    parser.add_argument('--ssp', type=str, default=ssp_list, help='')
    parser.add_argument('--var', type=str, default=var_list, help='')
    parser.add_argument('--time_aggregation', type=str, default=time_aggr, help='')
    parser.add_argument('--agg_files_dir', type=str, default=aggregated_files_path, help='')
    pars = vars(parser.parse_args())

    isimip_path = pars['isimip_path']
    ssp_list = pars['ssp'].split('+')
    var_list = pars['var'].split('+')
    time_aggr = pars['time_aggregation']
    aggregated_files_path = pars['agg_files_dir']

    parallel_aggregation(var_list, ssp_list)

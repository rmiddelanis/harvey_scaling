import argparse
import itertools
import os
import pickle
import random
from shapely.geometry import Point
from shapely.ops import unary_union
import numpy as np
import geopandas as gpd
import pyproj
import shapely.ops as ops
from shapely.geometry.polygon import Polygon
from functools import partial
import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool

base_dir = "{}/repos/harvey_scaling/sst_gmt_relationship".format(os.path.expanduser('~'))
aggregated_files_path = os.path.join(base_dir, "data/isimip/aggregated/")
# land_shares_path = os.path.join(base_dir, "data/land_share_0.5x0.5_lon:-179.75:179.75_lat:-89.75:89.75.geojson")
global_land_shares_path = os.path.join(base_dir, "data/land_share_geojson/land_share_world_0.5x0.5_lon:-179.25:179.25_lat:-89.25:89.25.geojson")
sea_land_shares_path = os.path.join(base_dir, "data/land_share_geojson/land_share_{}_0.5x0.5_lon:-179.75:179.75_lat:-89.75:89.75.geojson")
world_shapefile_path = "/home/robin/data/GADM/world_gadm36_levels_shp/gadm36_0.shp"
world_shapefile_union_path = "{}/repos/harvey_scaling/sst_gmt_relationship/data/shapes/world_shape_union_complete_epsg4326.pk".format(
    os.path.expanduser('~'))
world_shapefile_union_simplified_eck4_path = "{}/repos/harvey_scaling/sst_gmt_relationship/data/shapes/world_shape_union_complete_simplified_eck4.pk".format(
    os.path.expanduser('~'))
shapefile_template_eck4 = "{}/repos/harvey_scaling/sst_gmt_relationship/data/shapes/{}_eck4.pk".format(
    os.path.expanduser('~'), '{}')
ssp_list = 'ssp126+ssp370+ssp585'
var_list = 'tas'
time_aggr = 'monthly'
lat_nor = 23.5
lat_sou = -23.5


def transform_shape(shp, _from: str, _to: str):
    shp_trf = ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init=_from),
            pyproj.Proj(proj=_to, )
        ),
        shp)
    return shp_trf


def generate_world_shape_union(_lon_min=None, _lon_max=None, _lat_min=None, _lat_max=None, _ctry_selection=None):
    if _lon_min is None:
        _lon_min = -180
    if _lon_max is None:
        _lon_max = 180
    if _lat_min is None:
        _lat_min = -90
    if _lat_max is None:
        _lat_max = 90
    world_shape = gpd.read_file(world_shapefile_path)
    if _ctry_selection is not None:
        world_shape = world_shape[world_shape['NAME_0'].isin(_ctry_selection)]
    lon_lat_selection = [True] * len(world_shape)
    for idx in range(len(world_shape)):
        bounds = world_shape.iloc[idx].geometry.bounds
        if bounds[0] > _lon_max or bounds[2] < _lon_min or bounds[1] > _lat_max or bounds[3] < _lat_min:
            lon_lat_selection[idx] = False
    world_shape_union = unary_union(world_shape[lon_lat_selection].geometry.values)
    return world_shape_union


def calc_cell_land_share(args):
    if len(args) == 5:
        (lonlats, lat_res, lon_res, shape_path, shape_type) = args
    else:
        raise ValueError('must provide (lonlats, _lat_res, _lon_res, _world_shape). args has len {}'.format(len(args)))
    if shape_type not in ['sea', 'land']:
        raise ValueError('shape type must be either \'land\' or \'sea\'.)')
    geo_shape = pickle.load(open(shape_path, 'rb'))
    land_shares = []
    areas = []
    for (lon, lat) in tqdm.tqdm(lonlats):
        cell_points = [(lon + lon_res / 2, lat + lat_res / 2), (lon + lon_res / 2, lat - lat_res / 2),
                       (lon - lon_res / 2, lat - lat_res / 2), (lon - lon_res / 2, lat + lat_res / 2)]
        cell_geometry = Polygon(cell_points)
        cell_geometry = ops.transform(
            partial(
                pyproj.transform,
                pyproj.Proj(init='EPSG:4326'),
                pyproj.Proj(proj='eck4')
            ),
            cell_geometry)
        cell_area = cell_geometry.area
        intersection_share = geo_shape.intersection(cell_geometry).area / cell_area
        if shape_type == 'land':
            land_shares.append(intersection_share)
        elif shape_type == 'sea':
            land_shares.append(1 - intersection_share)
        areas.append(cell_area)
    return lonlats, areas, land_shares


def calc_grid_weights(_region_name, _lon_min=None, _lon_max=None, _lat_min=None, _lat_max=None, _lat_res=0.5,
                      _lon_res=0.5, _ctry_selection=None):
    if _lon_min is None:
        _lon_min = -180 + _lon_res / 2
    if _lon_max is None:
        _lon_max = 180 - _lon_res / 2
    if _lat_min is None:
        _lat_min = -90 + _lat_res / 2
    if _lat_max is None:
        _lat_max = 90 - _lat_res / 2
    lats = np.arange(_lat_min, min(_lat_max + _lat_res / 2, 90), _lat_res)
    lons = np.arange(_lon_min, min(_lon_max + _lon_res / 2, 180), _lon_res)
    _lat_max = max(lats)
    _lon_max = max(lons)
    filename = 'land_share_{}_{}x{}_lon:{}:{}_lat:{}:{}.geojson'.format(_region_name, _lon_res, _lat_res, _lon_min,
                                                                        _lon_max, _lat_min, _lat_max)
    if os.path.exists(os.path.join(base_dir, "data/land_share_geojson/{}".format(filename))):
        print('file {} already exists. Doing nothing.'.format(filename))
    else:
        print('calculating cell weight for {} cells'.format(int(len(lats) * len(lons))))

    pool = Pool()

    num_workers = multiprocessing.cpu_count()

    lonlat_tuples = list(itertools.product(lons, lats))
    random.shuffle(lonlat_tuples)
    lonlats = []
    stepwidth = int(len(lonlat_tuples) / num_workers)

    for i in range(num_workers - 1):
        lonlats.append(lonlat_tuples[i * stepwidth:(i + 1) * stepwidth])
    lonlats.append(lonlat_tuples[(i + 1) * stepwidth:])

    if _region_name == 'world':
        _shape_path = world_shapefile_union_simplified_eck4_path
        _shape_type = 'land'
    elif _region_name in ['arctic_ocean', 'indian_ocean', 'north_atlantic_ocean', 'south_atlantic_ocean',
                          'southern_ocean', 'south_pacific_ocean']:
        _shape_path = shapefile_template_eck4.format(_region_name)
        _shape_type = 'sea'
    else:
        raise ValueError('Must provice valid region name.')
    res = pool.map(calc_cell_land_share, zip(lonlats,
                                             itertools.repeat(_lat_res),
                                             itertools.repeat(_lon_res),
                                             itertools.repeat(_shape_path),
                                             itertools.repeat(_shape_type)
                                             ))
    land_shares = []
    areas = []
    lons = []
    lats = []
    for worker_res in res:
        lons_lats = list(zip(*worker_res[0]))
        lons += lons_lats[0]
        lats += lons_lats[1]
        areas += worker_res[1]
        land_shares += worker_res[2]
    land_share_gdf = gpd.GeoDataFrame()
    land_share_gdf['lon'] = lons
    land_share_gdf['lat'] = lats
    land_share_gdf['area'] = areas
    land_share_gdf['land_share'] = land_shares
    land_share_gdf.geometry = land_share_gdf.apply(lambda x: Point(x['lon'], x['lat']), axis=1)
    land_share_gdf.to_file(os.path.join(base_dir, 'data/{}'.format(filename)), driver='GeoJSON')


def calc_grid_sizes(gdf):
    lats = sorted(gdf['lat'].unique())
    lons = sorted(gdf['lon'].unique())
    lat_res = lats[1] - lats[0]
    lon_res = lons[1] - lons[0]
    cell_sizes = {}
    lon = 0
    for lat in lats:
        cell_points = [(lon + lon_res / 2, lat + lat_res / 2), (lon + lon_res / 2, lat - lat_res / 2),
                       (lon - lon_res / 2, lat - lat_res / 2), (lon - lon_res / 2, lat + lat_res / 2)]
        cell_geometry = Polygon(cell_points)
        cell_geometry = ops.transform(
            partial(
                pyproj.transform,
                pyproj.Proj(init='EPSG:4326'),
                pyproj.Proj(proj='eck4')
            ),
            cell_geometry)
        cell_sizes[lat] = cell_geometry.area
    gdf['area'] = gdf['lat'].apply(lambda x: cell_sizes[x])


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('--ssp', type=str, default=ssp_list, help='')
#     parser.add_argument('--var', type=str, default=var_list, help='')
#     parser.add_argument('--time_aggregation', type=str, default=time_aggr, help='')
#     parser.add_argument('--northern_lat', type=float, default=lat_nor, help='')
#     parser.add_argument('--southern_lat', type=float, default=lat_sou, help='')
#     parser.add_argument('--agg_files_dir', type=str, default=aggregated_files_path, help='')
#     pars = vars(parser.parse_args())
#
#     ssp_list = pars['ssp'].split('+')
#     var_list = pars['var'].split('+')
#     time_aggr = pars['time_aggregation']
#     lat_nor = pars['northern_lat']
#     lat_sou = pars['southern_lat']
#     aggregated_files_path = pars['agg_files_dir']

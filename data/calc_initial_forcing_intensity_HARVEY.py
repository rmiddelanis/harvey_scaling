import shapely.geometry as geometry
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
from descartes import PolygonPatch
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
import numpy as np
import math
from tqdm import tqdm
import json
import os

rootdir = os.path.join(os.getenv("HOME"), 'repos/harvey_scaling/')

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
mpl.rcParams['figure.dpi'] = 100

storm = 'HARVEY'
states = ['TX', 'LA']
economic_year = 2015
hazard_year = 2017
# radius_extensions = [0, 100e3, 200e3, 300e3]
# radius_extensions = [10000 + i * 10000 for i in range(10)]
radius_extensions = [0 + i * 10000 for i in range(11)]
# radius_extensions = np.arange(0, 100000 + 1, 10000)
alpha = 1.87
baseline_nc_path = os.path.join(rootdir, 'data/external/EORA_{}_baseline/output.nc'.format(economic_year))
county_gdp_path = os.path.join(rootdir, 'data/external/all_counties_GDP_2012_2015_2017.csv')
hwm_path = os.path.join(rootdir, 'data/external/HARVEY_high_water_marks.csv')


def plot_polygon(polygon, ax, _fc='#999999', _ec='k'):
    patch = PolygonPatch(polygon, fc=_fc, ec=_ec, fill=True, alpha=0.3)
    ax.add_patch(patch)

def alpha_shape(_points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param _points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(_points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(_points)).convex_hull

    def add_edge(_edges, _edge_points, _coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in _edges or (j, i) in _edges:
            # already added
            return
        _edges.add((i, j))
        _edge_points.append(_coords[[i, j]])

    coords = np.array([point.coords[0] for point in _points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        # Semiperimeter of triangle
        s = (a + b + c) / 2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        # Here's the radius filter.
        # print circum_r
        if circum_r < 1.0e6 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points


def linreg(_x, _y):
    _x = np.asarray(_x)
    _y = np.asarray(_y)
    _A = np.vstack([_x, np.ones(len(_x))]).T
    return np.linalg.lstsq(_A, _y, rcond=None)[0]


def load_hwm():
    hwm_df = pd.read_csv(hwm_path)
    points = gpd.points_from_xy(hwm_df.longitude_dd, hwm_df.latitude_dd)
    hwm_gdf = gpd.geodataframe.GeoDataFrame(hwm_df, geometry=points)
    hwm_gdf.set_crs(epsg=4326, inplace=True)
    hwm_gdf.to_crs(epsg=3663, inplace=True)
    return hwm_gdf


if __name__ == '__main__':
    us_states_shp = gpd.read_file(os.path.join(rootdir, 'data/external/gadm36_USA_shp/gadm36_USA_1.shp')).to_crs(epsg=3663)
    us_county_shp = gpd.read_file(os.path.join(rootdir, 'data/external/gadm36_USA_shp/gadm36_USA_2.shp')).to_crs(epsg=3663)
    us_county_shp.drop(us_county_shp[us_county_shp['HASC_2'].isna()].index, inplace=True)
    lasalle_idx = us_county_shp[(us_county_shp['NAME_2'] == 'La Salle') & (us_county_shp['NAME_1'] == 'Louisiana')].index[0]
    us_county_shp.loc[lasalle_idx, 'NAME_2'] = 'LaSalle'
    us_county_shp['state'] = us_county_shp['HASC_2'].apply(lambda x: x[3:5])
    us_states_shp['state'] = us_states_shp['HASC_1'].apply(lambda x: x[3:5])
    us_county_shp['county'] = us_county_shp['NAME_2'].apply(lambda x: x.lower())
    us_states_shp = us_states_shp[us_states_shp['NAME_1'].isin(['Louisiana', 'Texas'])]
    us_county_shp = us_county_shp[us_county_shp['NAME_1'].isin(['Louisiana', 'Texas'])]

    hwm_gdf = load_hwm()

    concave_hull, _ = alpha_shape(hwm_gdf.geometry, alpha=alpha)

    county_gdp_df = pd.read_csv(county_gdp_path, header=4, na_values='(NA)')
    county_gdp_df.dropna(inplace=True)
    county_gdp_df['state'] = county_gdp_df['GeoName'].apply(lambda x: x.split(',')[-1][1:])
    county_gdp_df['county'] = county_gdp_df['GeoName'].apply(lambda x: ','.join(x.split(',')[:-1]))
    county_gdp_df['county'] = county_gdp_df['county'].apply(lambda x: 'Saint' + x[3:] if x[:3] == 'St.' else x)
    county_gdp_df['county'] = county_gdp_df['county'].apply(lambda x: x.lower())

    exposed_gdp = {}
    affected_counties = {}
    for re in radius_extensions:
        exposed_gdp[re] = {}
        affected_counties[re] = []
        for st in states:
            exposed_gdp[re][st] = 0

    added_counties = []
    for re in tqdm(radius_extensions):
        concave_hull_ext = concave_hull.buffer(re)
        for idx, county_row in us_county_shp.iterrows():
            if county_row.geometry.intersects(concave_hull_ext):
                if idx not in added_counties:
                    added_counties.append(idx)
                    affected_counties[re].append(county_row.HASC_2)
                county_gdp = county_gdp_df[(county_gdp_df['county'] == county_row.county) & (county_gdp_df['state'] == county_row.state)][str(economic_year)]
                if len(county_gdp) == 1:
                    exposed_gdp[re][county_row.state] += county_gdp.iloc[0]
                elif len(county_gdp) > 1:
                    print('Warning. More than one value for county {} in state {} and year {}'.format(county_row.NAME_2,
                                                                                                      county_row.state,
                                                                                                      economic_year))
                    print(county_gdp)
                else:
                    print('Warning. No GDP value found for county {} in state {} and year {}'.format(county_row.NAME_2,
                                                                                                     county_row.state,
                                                                                                     economic_year))
    with open(os.path.join(rootdir, 'data/generated/affected_counties.json'), 'w') as f:
        json.dump(affected_counties, f)

    initial_forcing_intensities = {'points': {}, 'params': {}}
    for re in radius_extensions:
        initial_forcing_intensities['points'][re] = {}
        for st in states:
            initial_forcing_intensities['points'][re][st] = exposed_gdp[re][st] / county_gdp_df[county_gdp_df['state'] == st].sum()[str(economic_year)]

    x = radius_extensions
    for state in states:
        y_f0 = [initial_forcing_intensities['points'][re][state] for re in radius_extensions]
        m_f0, c_f0 = linreg(x, y_f0)
        initial_forcing_intensities['params'][state] = {'m': m_f0, 'c': c_f0}

    with open(os.path.join(rootdir, 'data/generated/initial_forcing_params.json'), 'w') as f:
        json.dump(initial_forcing_intensities, f)

import pandas as pd
import os
import us
import pycountry as pc

home_dir = os.getenv("HOME")

year = 2015
us_state_abbr_mapping = us.states.mapping('name', 'abbr')

naics_eora_mapping = pd.read_csv(os.path.join(home_dir, 'repos/harvey_scaling/data/generated/us_disaggregation/naics_eora_mapping.csv'))

gdp = pd.read_csv(os.path.join(home_dir, 'repos/harvey_scaling/data/external/SAGDP2N__ALL_AREAS_1997_2020.csv'),
                  engine='python', na_values=['(NA)', '(D)'])
gdp = gdp[['GeoName', 'LineCode', str(year)]]
gdp[str(year)] = gdp[str(year)].apply(lambda x: x if x != '(L)' else 0)  # (L) = values below 50k$ --> set to 0
gdp = gdp[gdp['GeoName'].isin(us_state_abbr_mapping.keys())]
gdp = gdp.dropna()
gdp = gdp.astype({'GeoName': str, 'LineCode': int, str(year): float})
gdp['GeoName'] = gdp['GeoName'].apply(lambda x: 'US.' + us_state_abbr_mapping[x])
gdp = gdp[gdp['LineCode'].isin(naics_eora_mapping['BEA_line_code'].unique())]
gdp = gdp.rename({'GeoName': 'region', 'LineCode': 'sector', '{}'.format(year): 'gdp'}, axis=1)

gdp.to_csv(os.path.join(home_dir, 'repos/harvey_scaling/data/generated/us_disaggregation/state_gdp_naics_{}.csv'.format(year)), sep=',')
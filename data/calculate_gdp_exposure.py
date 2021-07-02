import pandas as pd

# Data from: https://apps.bea.gov/regional/downloadzip.cfm
# CAGDP2: GDP in Current Dollars by Counties and MSA
county_gdp = pd.read_csv("external/CAGDP2__ALL_AREAS_2001_2019.csv", engine='python', na_values='(NA)')
county_gdp = county_gdp[county_gdp['Description'] == 'All industry total']
county_gdp = county_gdp[['GeoName', '2017']]
county_gdp = county_gdp[county_gdp['GeoName'].apply(lambda x: ', ' in x)]
county_gdp['GeoName'] = county_gdp['GeoName'].apply(lambda x: x[:-1] if x[-1] == '*' else x)
county_gdp['State'] = county_gdp['GeoName'].apply(lambda x: x[-2:])
county_gdp['County'] = county_gdp['GeoName'].apply(lambda x: x[:-4])
county_gdp.drop('GeoName', axis=1, inplace=True)
county_gdp.set_index(['State', 'County'], drop=True, inplace=True)
county_gdp.dropna(inplace=True)
county_gdp = county_gdp.astype(int)

state_gdp = county_gdp.groupby('State').sum()
state_gdp.rename({'2017': 'gdp'}, inplace=True, axis=1)

# High water Mark data from: https://stn.wim.usgs.gov/FEV/#Sandy
hwm = pd.read_csv("external/HARVEY_high_water_marks.csv")[['stateName', 'countyName']]
hwm.rename({'stateName': 'State', 'countyName': 'County'}, axis=1, inplace=True)
hwm['County'] = hwm['County'].apply(lambda x: x[:-7])
hwm.drop_duplicates(inplace=True)
exposed_gdp = hwm.set_index(['State', 'County']).join(county_gdp).groupby('State').sum()
exposed_gdp.rename({'2017': 'exposed_gdp'}, inplace=True, axis=1)
exposed_gdp = exposed_gdp.join(state_gdp)
exposed_gdp['exposure'] = exposed_gdp['exposed_gdp'] / exposed_gdp['gdp']
exposed_gdp.to_csv('generated/gdp_exposure.csv', sep=',', na_rep=[''])

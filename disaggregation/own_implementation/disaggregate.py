import argparse
from netCDF4 import Dataset
import numpy as np
import pandas as pd


def main():
    if disagg_region == 'USA':
        state_gdp_naics_path = "/home/robin/repos/harvey_scaling/disaggregation/own_implementation/us_state_gdp_naics_2015.csv"
        state_gdp_path = "/home/robin/repos/harvey_scaling/disaggregation/own_implementation/us_state_gdp_2015.csv"
        naics_eora_mapping_path = "/home/robin/repos/harvey_scaling/disaggregation/own_implementation/naics_eora_mapping.csv"
        sub_regions = ["US.AL", "US.AK", "US.AZ", "US.AR", "US.CA", "US.CO", "US.CT", "US.DE", "US.DC",
                       "US.FL", "US.GA", "US.HI", "US.ID", "US.IL", "US.IN", "US.IA", "US.KS", "US.KY",
                       "US.LA", "US.ME", "US.MD", "US.MA", "US.MI", "US.MN", "US.MS", "US.MO", "US.MT",
                       "US.NE", "US.NV", "US.NH", "US.NJ", "US.NM", "US.NY", "US.NC", "US.ND", "US.OH",
                       "US.OK", "US.OR", "US.PA", "US.RI", "US.SC", "US.SD", "US.TN", "US.TX", "US.UT",
                       "US.VT", "US.VA", "US.WA", "US.WV", "US.WI", "US.WY"]
        subregion_gdp = pd.read_csv(state_gdp_path, index_col='region')
        state_gdp_naics = pd.read_csv(state_gdp_naics_path, index_col=['region', 'sector'])
        naics_eora_mapping = pd.read_csv(naics_eora_mapping_path)
        naics_eora_mapping = naics_eora_mapping[naics_eora_mapping['has_child'] == 0][['BEA_line_code', 'EORA_sector']]
        naics_eora_mapping = pd.DataFrame(naics_eora_mapping.groupby('EORA_sector')['BEA_line_code'].apply(list))
        naics_eora_mapping['factors'] = naics_eora_mapping['BEA_line_code'].apply(lambda x: [1] * len(x))
        eora_naics_lookup = {}
        for mapping_idx in naics_eora_mapping.index:
            if len(mapping_idx) > 4:
                split_sectors = mapping_idx.split('+')
                split_sector_line_codes = naics_eora_mapping.loc[mapping_idx, 'BEA_line_code']
                split_sector_factors = [1 / len(split_sectors)] * len(split_sector_line_codes)
                for split_sector in split_sectors:
                    if split_sector in list(eora_naics_lookup.keys()):
                        eora_naics_lookup[split_sector]['BEA_line_code'] += split_sector_line_codes
                        eora_naics_lookup[split_sector]['factors'] += split_sector_factors
                    else:
                        eora_naics_lookup[split_sector] = {'BEA_line_code': split_sector_line_codes,
                                                           'factors': split_sector_factors}
            else:
                eora_naics_lookup[mapping_idx] = {'BEA_line_code': naics_eora_mapping.loc[mapping_idx, 'BEA_line_code'],
                                                  'factors': naics_eora_mapping.loc[mapping_idx, 'factors']}
    elif disagg_region == 'CHN':
        sub_regions = ["CN.AH", "CN.BJ", "CN.CQ", "CN.FJ", "CN.GS", "CN.GD", "CN.GX", "CN.GZ", "CN.HA", "CN.HB",
                       "CN.HL", "CN.HE", "CN.HU", "CN.HN", "CN.JS", "CN.JX", "CN.JL", "CN.LN", "CN.NM", "CN.NX",
                       "CN.QH", "CN.SA", "CN.SD", "CN.SH", "CN.SX", "CN.SC", "CN.TJ", "CN.XJ", "CN.XZ", "CN.YN",
                       "CN.ZJ"]
        chn_gdp = pd.read_csv("/home/robin/repos/harvey_scaling/disaggregation/own_implementation/chn_gdp.csv",
                              index_col='id')
        subregion_gdp = chn_gdp[chn_gdp['year'] == 2015]
        eora_naics_lookup = {}

    in_dataset = Dataset(in_path, 'r', format='NETCDF-4')
    in_sectors = in_dataset['sector'][:]
    in_regions = in_dataset['region'][:]

    disagg_reg_indices = np.where(in_regions == disagg_region)[0]
    if len(in_dataset['flows'].shape) == 4:
        disagg_reg_flow_indices = disagg_reg_indices
    elif len(in_dataset['flows'].shape) == 2:
        disagg_reg_flow_indices = np.where(in_dataset['index_region'] == np.where(in_regions == disagg_region)[0][0])[0]

    out_regions = np.concatenate([in_regions[:disagg_reg_indices[0]], sub_regions, in_regions[disagg_reg_indices[-1] + 1:]])
    out_sectors = in_sectors

    out_sector_indices = {}
    for out_sec_index, out_sec in enumerate(out_sectors):
        out_sector_indices[out_sec] = out_sec_index

    out_region_indices = {}
    for out_reg_index, out_reg in enumerate(out_regions):
        out_region_indices[out_reg] = out_reg_index

    gdp_shares = pd.DataFrame(columns=['region', 'sector', 'gdp'])
    for region in sub_regions:
        for sector in out_sectors:
            if sector in eora_naics_lookup.keys():
                gdp = 0
                for line_code, factor in zip(*tuple(eora_naics_lookup[sector].values())):
                    gdp += state_gdp_naics.loc[(region, line_code), 'gdp'] * factor
            else:
                gdp = subregion_gdp.loc[region, 'gdp']
            gdp_shares.loc[len(gdp_shares)] = [region, sector, gdp]
    gdp_shares['gdp_share'] = gdp_shares.apply(
        lambda x: x['gdp'] / gdp_shares.groupby('sector')['gdp'].sum()[x['sector']], axis=1)
    gdp_shares = gdp_shares.set_index(['region', 'sector'])['gdp_share'].unstack().loc[sub_regions, in_sectors]

    with Dataset(out_path, 'w') as outfile:
        sector_dim = outfile.createDimension("sector", len(out_sectors))
        region_dim = outfile.createDimension("region", len(out_regions))
        index_dim = outfile.createDimension("index", len(out_sectors) * len(out_regions))

        sector_var = outfile.createVariable("sector", str, "sector")
        sector_var[:] = out_sectors

        region_var = outfile.createVariable("region", str, "region")
        region_var[:] = out_regions

        index_sector_var = outfile.createVariable("index_sector", int, "index")
        index_region_var = outfile.createVariable("index_region", int, "index")
        flows_var = outfile.createVariable("flows", float, ("index", "index"), zlib=True, complevel=7)

        index_region_var[:] = [i for i in range(len(out_regions)) for j in range(len(out_sectors))]
        index_sector_var[:] = list(range(len(out_sectors))) * len(out_regions)

        temp = np.ma.masked_all(flows_var[:].shape)
        if len(in_dataset['flows'].shape) == 4:
            unchanged_flows = np.delete(np.delete(in_dataset['flows'][:], disagg_reg_flow_indices, axis=1), disagg_reg_flow_indices,
                                        axis=3)
            unchanged_flows = unchanged_flows.transpose().reshape(
                (len(in_regions) - 1, len(in_sectors), (len(in_regions) - 1) * len(in_sectors))).transpose(2, 0, 1).reshape(
                ((len(in_regions) - 1) * len(in_sectors), (len(in_regions) - 1) * len(in_sectors)))
        elif len(in_dataset['flows'].shape) == 2:
            unchanged_flows = np.delete(np.delete(in_dataset['flows'][:], disagg_reg_flow_indices, axis=0),
                                        disagg_reg_flow_indices,
                                        axis=1)
        subregion_indices = np.where(np.isin(out_regions, sub_regions))
        unchanged_region_slice = ~np.isin(index_region_var[:], subregion_indices)
        unchanged_flows_slice = np.repeat(unchanged_region_slice.reshape(-1, 1), len(unchanged_region_slice), axis=1)
        unchanged_flows_slice = unchanged_flows_slice & unchanged_flows_slice.transpose()
        temp[unchanged_flows_slice] = unchanged_flows[
            np.array([[True] * unchanged_flows.shape[1]] * unchanged_flows.shape[0])]

        if len(in_dataset['flows'].shape) == 4:
            old_outflows = np.delete(in_dataset['flows'][:, disagg_reg_flow_indices[0]:disagg_reg_flow_indices[-1] + 1, ...],
                                     disagg_reg_flow_indices, axis=3)
            new_outflows = old_outflows * np.ones((len(in_sectors), len(sub_regions), len(in_sectors), len(in_regions) - 1))
            new_outflows = (new_outflows.transpose() * gdp_shares.values).transpose()
            new_outflows = new_outflows.transpose().reshape(
                (len(in_regions) - 1, len(in_sectors), len(sub_regions) * len(in_sectors))).transpose(2, 0, 1).reshape(
                (len(sub_regions) * len(in_sectors), (len(in_regions) - 1) * len(in_sectors)))
        elif len(in_dataset['flows'].shape) == 2:
            old_outflows = np.delete(in_dataset['flows'][disagg_reg_flow_indices[0]:disagg_reg_flow_indices[-1] + 1, :], disagg_reg_flow_indices, axis=1)
            new_outflows = np.repeat(np.expand_dims(old_outflows, 1), len(sub_regions), axis=1)
            new_outflows = (new_outflows.transpose() * gdp_shares.values).transpose()
            new_outflows = new_outflows.transpose(2, 0, 1).reshape((len(in_regions) - 1) * len(in_sectors), (len(sub_regions) * len(in_sectors))).transpose()
        subregion_slice = ~unchanged_region_slice
        changed_outflows_slice = np.repeat(subregion_slice.reshape(-1, 1), temp.shape[1], axis=1) & np.repeat(
            unchanged_region_slice.reshape(-1, 1), temp.shape[1], axis=1).transpose()
        temp[changed_outflows_slice] = new_outflows[np.array([[True] * new_outflows.shape[1]] * new_outflows.shape[0])]

        if len(in_dataset['flows'].shape) == 4:
            old_inflows = np.delete(in_dataset['flows'][..., disagg_reg_flow_indices[0]:disagg_reg_flow_indices[-1] + 1],
                                    disagg_reg_flow_indices, axis=1)
            new_inflows = old_inflows * np.ones((len(in_sectors), len(in_regions) - 1, len(in_sectors), len(sub_regions)))
            new_inflows = new_inflows * gdp_shares.values.transpose()
            new_inflows = new_inflows.transpose().reshape(
                (len(sub_regions), len(in_sectors), (len(in_regions) - 1) * len(in_sectors))).transpose(2, 0, 1).reshape(
                ((len(in_regions) - 1) * len(in_sectors), len(sub_regions) * len(in_sectors)))
        elif len(in_dataset['flows'].shape) == 2:
            old_inflows = np.delete(in_dataset['flows'][:, disagg_reg_flow_indices[0]:disagg_reg_flow_indices[-1] + 1], disagg_reg_flow_indices, axis=0)
            new_inflows = np.repeat(np.expand_dims(old_inflows, 2), len(sub_regions), axis=2)
            new_inflows = new_inflows * gdp_shares.values.transpose()
            new_inflows = new_inflows.transpose(0, 2, 1).reshape((new_inflows.shape[0], len(sub_regions) * len(in_sectors)))
        changed_inflows_slice = changed_outflows_slice.transpose()
        temp[changed_inflows_slice] = new_inflows[np.array([[True] * new_inflows.shape[1]] * new_inflows.shape[0])]

        if len(in_dataset['flows'].shape) == 4:
            old_self_supply = in_dataset['flows'][:, disagg_reg_flow_indices[0]:disagg_reg_flow_indices[-1] + 1, :, disagg_reg_flow_indices[0]:disagg_reg_flow_indices[-1] + 1]
            new_self_supply = old_self_supply * np.ones((len(in_sectors), len(sub_regions), len(in_sectors), len(sub_regions)))
        elif len(in_dataset['flows'].shape) == 2:
            old_self_supply = in_dataset['flows'][disagg_reg_flow_indices[0]:disagg_reg_flow_indices[-1] + 1, disagg_reg_flow_indices[0]:disagg_reg_flow_indices[-1] + 1]
            new_self_supply = np.repeat(np.repeat(np.expand_dims(old_self_supply, (1, 3)), len(sub_regions), axis=1), len(sub_regions), axis=3)
        new_self_supply = ((new_self_supply * gdp_shares.values.transpose()).transpose() * gdp_shares.values).transpose()
        new_self_supply = new_self_supply.transpose().reshape(len(sub_regions), len(in_sectors), len(sub_regions) * len(in_sectors)).transpose(2, 0, 1).reshape((len(sub_regions) * len(in_sectors), len(sub_regions) * len(in_sectors)))
        changed_self_supply_slice = np.repeat(subregion_slice.reshape(-1, 1), temp.shape[1], axis=1) & np.repeat(
            subregion_slice.reshape(-1, 1), temp.shape[1], axis=1).transpose()
        temp[changed_self_supply_slice] = new_self_supply[
            np.array([[True] * new_self_supply.shape[1]] * new_self_supply.shape[0])]

        flows_var[:] = temp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--disagg_region', type=str, default="USA")
    parser.add_argument('--in_path', type=str, default="/home/robin/data/EORA/Eora26-v199.82-2015.nc")
    parser.add_argument('--out_path', type=str, default='')
    pars = vars(parser.parse_args())

    in_path = pars['in_path']
    disagg_region = pars['disagg_region']
    if pars['out_path'] != '':
        out_path = pars['out_path']
    else:
        out_path = "/home/robin/repos/harvey_scaling/disaggregation/own_implementation/output/" + in_path.split('/')[
            -1].replace('.nc', '_{}.nc'.format(disagg_region))

    main()

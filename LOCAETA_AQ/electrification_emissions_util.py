import pandas as pd
import geopandas as gpd
import os
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt


class ElectrificationEmissionProcessor:
    """
    A class to handle Data Center emission data processing and NEI integration.
    """
    
    def __init__(self, config):
        """
        Initialize the processor with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with file paths and parameters
        """
        self.config = config
        self.NEI_cols = ['NOx','SOx', 'NH3', 'VOC', 'PM2_5']

    def reformat_powerplant(self, df):

        # columns I need
        final_cols = [col for col in df.columns if '_tons_final' in col]
        base_cols = [col for col in df.columns if '_tons_base' in col]
        Facilities_col_names = final_cols + base_cols + ['eis','cambium_gea','DOE/EIA ORIS plant or facility code']

        # subset the dataframe 
        df = df[Facilities_col_names] 

        # Total before grouping
        total_before = df[final_cols + base_cols].sum()

        # drop the rows if eis is missing
        df = df.dropna(subset=['eis'])

        # Total after grouping
        total_after = df[final_cols+ base_cols].sum()

        # Define columns as integers
        df = df.astype({'eis': 'int64', 'DOE/EIA ORIS plant or facility code': 'int64'})

        # rename columns
        df.rename(columns={'eis': 'EIS_ID', 'DOE/EIA ORIS plant or facility code': "oris_ID"}, inplace=True)

        # Group by EIS_ID and aggregate emissions and cambium_gea
        df_grouped = df.groupby('EIS_ID').agg({
            **{col: 'sum' for col in final_cols+ base_cols},
            'cambium_gea': 'first' 
        }).reset_index()


        # Check if Totals are preserved
        print('before :', total_before, 'after: ', total_after) 
        #print("Reformatted df with cambium_gea:", df_grouped.head())

        return df_grouped

    from itertools import combinations

    def find_minimal_unique_identifier_columns(self, df, max_combination_size=30):
        """
        Finds the minimal set of columns that uniquely identify rows in a DataFrame.

        Args:
            df: pandas.DataFrame
            max_combination_size: int, maximum number of columns to consider in combinations (avoid long runtime)

        Returns:
            List of column names or None
        """
        cols = df.columns.tolist()
        for r in range(1, min(len(cols), max_combination_size) + 1):
            for combo in combinations(cols, r):
                if not df.duplicated(subset=combo).any():
                    return list(combo)
        return None

    def mapping_powerplant_to_nei(self, nei_with_powerplant, nei_all_pt, unique_identifier_columns, is_base):

        if is_base: 
            # Column mapping between NEI and eGRID
            pollutant_map = {
                'NOx': 'NOx_tons_base',
                'SOx': 'SO2_tons_base',
                'NH3': 'NH3_tons_base',
                'VOC': 'VOC_tons_base',
                'PM2_5': 'PM2.5_tons_base'}
        else:
            pollutant_map = {
                'NOx': 'NOx_tons_final',
                'SOx': 'SO2_tons_final',
                'NH3': 'NH3_tons_final',                                                
                'VOC': 'VOC_tons_final',
                'PM2_5': 'PM2.5_tons_final'}

        # add Boolean to track back the data center data later
        nei_with_powerplant['was_mapped'] = True  # add flag


        # Compute and apply split factors per pollutant
        for nei_col, Facilities_col in pollutant_map.items():

            print (nei_col, Facilities_col)
            # Group sum for each pollutant by EIS_ID
            total_by_eis = nei_with_powerplant.groupby('EIS_ID')[f'{nei_col}_nei'].transform('sum')
            nei_with_powerplant[f'{nei_col}_total_by_eis'] = total_by_eis

            # Default: compute split factor using NEI emissions
            split_col = f'{nei_col}_split'

            nei_with_powerplant[split_col] = np.where(
                total_by_eis == 0, 
                np.nan, 
                nei_with_powerplant[f'{nei_col}_nei'] / total_by_eis
            )

            # Find EIS_IDs where total_by_eis is zero but Facilities_col is non-zero
            mask_zero_total = (total_by_eis == 0) & nei_with_powerplant[Facilities_col].notna() & (nei_with_powerplant[Facilities_col] != 0)

            print(f"{nei_col}: # fallback allocations due to zero NEI = {mask_zero_total.sum()}")

            # For these EIS_IDs, assign equal split factor across matching rows
            for eid in nei_with_powerplant.loc[mask_zero_total, 'EIS_ID'].unique():
                match_rows = nei_with_powerplant['EIS_ID'] == eid
                n_rows = match_rows.sum()
                nei_with_powerplant.loc[match_rows, split_col] = 1.0 / n_rows

            # Now compute eGRID-scaled emissions and save as nei original name
            nei_with_powerplant[f'{nei_col}'] = nei_with_powerplant[split_col] * nei_with_powerplant[Facilities_col]

            print(f"after {Facilities_col} splitting : ", nei_with_powerplant[f'{nei_col}'].sum())
        # OPTIONAL: Drop intermediate split columns
        #nei_with_powerplant.drop(columns=[f'{k}_split' for k in pollutant_map], inplace=True)
        # Merge results back into the full NEI dataset

        # Merge results back into the full NEI dataset
        nei_all_pt_final = nei_all_pt.merge(
            nei_with_powerplant[ 
                unique_identifier_columns + ["was_mapped",'cambium_gea'] + [f'{k}' for k in pollutant_map]
            ],
            on=unique_identifier_columns,
            how='left'
        )

        gdf_subset = nei_all_pt_final[nei_all_pt_final['was_mapped'] == True]
        print("base dataframe size ", gdf_subset.shape, nei_with_powerplant.shape)
        print("before filling; subset nei sum ", gdf_subset[['PM2_5_nei', 'NH3_nei', 'VOC_nei', 'NOx_nei', 'SOx_nei']].sum())
        print("before filling; subset base sum ", gdf_subset[['PM2_5', 'NH3', 'VOC', 'NOx', 'SOx']].sum())
        print("before filling; nei_all_pt_final base sum ", nei_all_pt_final[['PM2_5', 'NH3', 'VOC', 'NOx', 'SOx']].sum()) 

        # fill the empty rows with NEI dataset
        for k in pollutant_map:
            nei_all_pt_final[f'{k}'] = nei_all_pt_final[f'{k}'].fillna(nei_all_pt_final[f'{k}_nei'])
            nei_all_pt_final[f'{k}_diff'] = nei_all_pt_final[f'{k}'] - nei_all_pt_final[f'{k}_nei']

        # Define difference columns
        diff_cols = ['NOx_diff', 'SOx_diff', 'NH3_diff',  'VOC_diff', 'PM2_5_diff']

        # Mask for rows that were mapped
        mapped_mask = nei_all_pt_final['was_mapped'] == True

        # Mask for rows with no difference in any pollutant
        no_change_mask = (nei_all_pt_final[diff_cols] == 0).all(axis=1)

        # Combine masks
        mapped_but_unchanged = nei_all_pt_final[mapped_mask & no_change_mask]

        # Show result
        print("Number of rows where emissions were mapped but did not change:", mapped_but_unchanged.shape[0])

        # Remove rows where all values in specified columns are zero
        gdf_subset = nei_all_pt_final[nei_all_pt_final['was_mapped'] == True]
        print("subset dataframe size ", gdf_subset.shape, nei_with_powerplant.shape)
        print("subset nei sum ", gdf_subset[['PM2_5_nei', 'NH3_nei', 'VOC_nei', 'NOx_nei', 'SOx_nei']].sum())
        print("subset base sum ", gdf_subset[['PM2_5', 'NH3', 'VOC', 'NOx', 'SOx']].sum())

        # drop the unnecessary columns
        nei_all_pt_final.drop(columns=[f'{k}_diff' for k in pollutant_map], inplace=True)
        #nei_all_pt_final.drop(columns=[f'{k}_nei' for k in pollutant_map], inplace=True)

        return nei_all_pt_final

    def diagnose_egrid_nei_mismatch(self, egrid, nei_all_pt, nei_with_powerplant):
        """
        Diagnose why eGRID sums don't match after filtering/merging with NEI data
        """
        print("=== eGRID-NEI EIS_ID MISMATCH DIAGNOSTIC ===")
        
        pollutant_cols = [ 'NOx_tons_base', 'SO2_tons_base','NH3_tons_base', 'VOC_tons_base', 'PM2.5_tons_base']
        
        # Get unique EIS_IDs from each dataset
        egrid_eids = set(egrid['EIS_ID'].unique())
        nei_eids = set(nei_all_pt['EIS_ID'].unique())
        merged_eids = set(nei_with_powerplant['EIS_ID'].unique())
        
        print(f"Unique EIS_IDs in eGRID: {len(egrid_eids)}")
        print(f"Unique EIS_IDs in NEI: {len(nei_eids)}")
        print(f"Unique EIS_IDs after merge: {len(merged_eids)}")
        
        # Find missing EIS_IDs
        egrid_not_in_nei = egrid_eids - nei_eids
        nei_not_in_egrid = nei_eids - egrid_eids
        egrid_lost_in_merge = egrid_eids - merged_eids
        
        print(f"\nEIS_IDs in eGRID but not in NEI: {len(egrid_not_in_nei)}")
        print(f"EIS_IDs in NEI but not in eGRID: {len(nei_not_in_egrid)}")
        print(f"EIS_IDs from eGRID lost after merge: {len(egrid_lost_in_merge)}")
        
        if len(egrid_not_in_nei) > 0:
            print(f"\n❌ FOUND THE PROBLEM: {len(egrid_not_in_nei)} eGRID facilities have no NEI data")
            
            # Calculate emissions lost due to missing EIS_IDs
            missing_egrid = egrid[egrid['EIS_ID'].isin(egrid_not_in_nei)]

            pollutant_cols = ['NOx_tons_final', 'SO2_tons_final','NH3_tons_final', 'VOC_tons_final', 'PM2.5_tons_final']
            
            print("\nFinal Emissions lost from missing EIS_IDs:")
            lost_emissions = missing_egrid[pollutant_cols].sum()
            for col, value in lost_emissions.items():
                print(f"  {col}: {value}") 


            
            print("\nBase Emissions lost from missing EIS_IDs:")
            lost_emissions = missing_egrid[pollutant_cols].sum()
            for col, value in lost_emissions.items():
                print(f"  {col}: {value}")



            print(f"\nAll missing EIS_IDs: {list(egrid_not_in_nei)}")
            
            # Show some details about the missing facilities
            print(f"\nDetails of first few missing facilities:")
            sample_missing = missing_egrid.head()[['EIS_ID'] + pollutant_cols]
            for idx, row in sample_missing.iterrows():
                print(f"  EIS_ID {row['EIS_ID']}: NOx={row['NOx_tons_base']}, PM2.5={row['PM2.5_tons_base']}")
        
        if len(egrid_lost_in_merge) > len(egrid_not_in_nei):
            print(f"\n❌ ADDITIONAL PROBLEM: More EIS_IDs lost in merge than expected")
            extra_lost = egrid_lost_in_merge - egrid_not_in_nei
            print(f"Extra lost EIS_IDs: {len(extra_lost)}")
            print(f"Sample extra lost: {list(extra_lost)[:5]}")
        
        # Verify the math
        original_sum = egrid[pollutant_cols].sum()
        kept_egrid = egrid[egrid['EIS_ID'].isin(merged_eids)]
        kept_sum = kept_egrid[pollutant_cols].sum()
        
        print(f"\n=== EMISSION ACCOUNTING ===")
        print("Original eGRID sums:")
        for col, val in original_sum.items():
            print(f"  {col}: {val}")
        
        print("\nSums for EIS_IDs that made it through merge:")
        for col, val in kept_sum.items():
            print(f"  {col}: {val}")
        
        print("\nDifference (lost emissions):")
        diff = original_sum - kept_sum
        for col, val in diff.items():
            print(f"  {col}: {val}")
        
        return {
            'egrid_not_in_nei': egrid_not_in_nei,
            'lost_emissions': missing_egrid[pollutant_cols].sum() if len(egrid_not_in_nei) > 0 else None,
            'kept_egrid': kept_egrid
        }

    # Quick function to check EIS_ID formats
    def check_eis_id_formats(self, egrid, nei_all_pt):
        """Check if EIS_ID formats might be causing mismatch"""
        print("=== EIS_ID FORMAT CHECK ===")
        
        egrid_sample = egrid['EIS_ID'].head(10).tolist()
        nei_sample = nei_all_pt['EIS_ID'].head(10).tolist()
        
        print("Sample eGRID EIS_IDs:", egrid_sample)
        print("Sample NEI EIS_IDs:", nei_sample)
        
        # Check data types
        print(f"\neGRID EIS_ID dtype: {egrid['EIS_ID'].dtype}")
        print(f"NEI EIS_ID dtype: {nei_all_pt['EIS_ID'].dtype}")
        
        # Check for leading/trailing spaces
        egrid_spaces = egrid['EIS_ID'].astype(str).str.contains('^ | $', regex=True).any()
        nei_spaces = nei_all_pt['EIS_ID'].astype(str).str.contains('^ | $', regex=True).any()
        
        if egrid_spaces or nei_spaces:
            print("❌ Found leading/trailing spaces in EIS_IDs")
            print(f"  eGRID has spaces: {egrid_spaces}")
            print(f"  NEI has spaces: {nei_spaces}")
        else:
            print("✅ No leading/trailing spaces found")
        
        # Check lengths
        egrid_lengths = egrid['EIS_ID'].astype(str).str.len().unique()
        nei_lengths = nei_all_pt['EIS_ID'].astype(str).str.len().unique()
        
        print(f"\neGRID EIS_ID lengths: {sorted(egrid_lengths)}")
        print(f"NEI EIS_ID lengths: {sorted(nei_lengths)}")


    def mapping_non_powerplant_to_nei(self, nei_with_non_powerplant, nei_df, unique_identifier_columns, is_base):

        if is_base: 
            pollutant_map = {
                'NOx': 'NOx_tons_base',
                'SOx': 'SO2_tons_base',
                'NH3': 'NH3_tons_base',
                'VOC': 'VOC_tons_base',
                'PM2_5': 'PM2.5_tons_base'}
        else:
            pollutant_map = {
                'NOx': 'NOx_tons_final',
                'SOx': 'SO2_tons_final',
                'NH3': 'NH3_tons_final',                                                
                'VOC': 'VOC_tons_final',
                'PM2_5': 'PM2.5_tons_final'}

        # add Boolean to track back the powerplant data later
        nei_with_non_powerplant['was_mapped'] = True  

        # Compute and apply split factors per pollutant
        for nei_col, powerplant_col in pollutant_map.items():
            print(nei_col, powerplant_col)

            # Group sum by (EIS_ID, SCC) now
            total_by_group = nei_with_non_powerplant.groupby(['EIS_ID', 'SCC'])[f'{nei_col}_nei'].transform('sum')

            # Default split factor within (EIS_ID, SCC)
            split_col = f'{nei_col}_split'
            nei_with_non_powerplant[split_col] = (
                nei_with_non_powerplant[f'{nei_col}_nei'] / total_by_group.replace(0, pd.NA)
            )

            # Handle zero-NEI cases where powerplant emissions exist
            mask_zero_total = (
                (total_by_group == 0) &
                nei_with_non_powerplant[powerplant_col].notna() &
                (nei_with_non_powerplant[powerplant_col] != 0)
            )

            print(f"{nei_col}: # fallback allocations due to zero NEI = {mask_zero_total.sum()}")

            for (eid, scc) in (
                nei_with_non_powerplant.loc[mask_zero_total, ['EIS_ID', 'SCC']]
                .drop_duplicates()
                .itertuples(index=False)
            ):
                match_rows = (nei_with_non_powerplant['EIS_ID'] == eid) & (nei_with_non_powerplant['SCC'] == scc)
                n_rows = match_rows.sum()
                nei_with_non_powerplant.loc[match_rows, split_col] = 1.0 / n_rows

            # Scale emissions and overwrite pollutant column
            nei_with_non_powerplant[nei_col] = (
                nei_with_non_powerplant[split_col] * nei_with_non_powerplant[powerplant_col]
            )

        # Merge results back into the full NEI dataset
        nei_all_pt_final = nei_df.merge(
            nei_with_non_powerplant[ 
                unique_identifier_columns + ["was_mapped", 'cambium_gea'] + [f'{k}' for k in pollutant_map]
            ],
            on=unique_identifier_columns,
            how='left'
        )

        gdf_subset = nei_all_pt_final[nei_all_pt_final['was_mapped'] == True]
        print("base dataframe size ", gdf_subset.shape, nei_with_non_powerplant.shape)
        print("before filling; subset nei sum ", gdf_subset[['PM2_5_nei', 'NH3_nei', 'VOC_nei', 'NOx_nei', 'SOx_nei']].sum())
        print("before filling; subset base sum ", gdf_subset[['PM2_5', 'NH3', 'VOC', 'NOx', 'SOx']].sum())
        print("before filling; nei_all_pt_final base sum ", nei_all_pt_final[['PM2_5', 'NH3', 'VOC', 'NOx', 'SOx']].sum()) 

        # Fill missing rows with NEI dataset values
        for k in pollutant_map:
            unmapped_mask = nei_all_pt_final['was_mapped'] != True
            nei_all_pt_final.loc[unmapped_mask, k] = nei_all_pt_final.loc[unmapped_mask, k].fillna(nei_all_pt_final.loc[unmapped_mask, f'{k}_nei'])

        return nei_all_pt_final 


    # Process powerplant emissions 
    def process_powerplant_scenario(self, scen_name, emis_name, nei_all_pt, config):
        """Process powerplant emissions for one scenario."""
        print("processing", scen_name)


        csv_input_dir = os.path.join(config["input"]["scenario_dir"], config["input"]["overall_scenario"])
        output_dir = os.path.join(config["output"]["output_dir"], config["input"]["overall_scenario"])
        os.makedirs(output_dir, exist_ok=True)

        facilities_file = os.path.join(csv_input_dir, f"pp_{scen_name}.csv")
        egrid = pd.read_csv(facilities_file)
        print("original data sum:\n", egrid.filter(like="_tons_base").sum())
        egrid = self.reformat_powerplant(egrid)
        print("after grouping sum:\n", egrid.filter(like="_tons_base").sum())

        # Filter NEI for relevant rows
        nei_with_pp = nei_all_pt[nei_all_pt["EIS_ID"].isin(egrid["EIS_ID"])].copy()
        nei_with_pp.drop(columns=["height", "diam", "temp", "velocity"], inplace=True)

        unique_id_cols = self.find_minimal_unique_identifier_columns(nei_with_pp)
        if unique_id_cols:
            print("Columns that uniquely identify rows:", unique_id_cols)
        else:
            print("No combination of columns uniquely identifies rows.")

        # Merge NEI + eGRID
        nei_with_pp = nei_with_pp.merge(egrid, on="EIS_ID", how="left")
        true_sum = nei_with_pp.groupby('EIS_ID')[['PM2.5_tons_base', 'NH3_tons_base', 'VOC_tons_base', 'NOx_tons_base', 'SO2_tons_base']].first().sum()
        print("intial nei_with_powerplant sum", true_sum)

        # Diagnostics
        # self.check_eis_id_formats(egrid, nei_all_pt)
        # diag = self.diagnose_egrid_nei_mismatch(egrid, nei_all_pt, nei_with_pp)
        # print(diag)

        # Process both base and final emissions
        for is_base in [True, False]:
            nei_all_pt_final = self.mapping_powerplant_to_nei(
                nei_with_pp, nei_all_pt, unique_id_cols, is_base=is_base
            )

            mapped_df = nei_all_pt_final[nei_all_pt_final['was_mapped'] == True].copy()
            unmapped_df = nei_all_pt_final[nei_all_pt_final['was_mapped'] != True].copy()

            print("after filling sum:\n", mapped_df[self.NEI_cols].sum())
            print("final size:", mapped_df.shape, unmapped_df.shape)

            suffix = "_pp_base" if is_base else "_pp"
            save_filename = os.path.join(output_dir, f"{emis_name}{suffix}.shp")
            if not mapped_df.empty:
                mapped_df.to_file(save_filename, driver="ESRI Shapefile")
                print(f"Saved mapped data to {save_filename}")

        return egrid, mapped_df, unmapped_df


    # process other facilities emissions
    def process_non_powerplant(self,  scen_emis_list, unmapped_df, nei_all_pt, config):

        csv_input_dir = os.path.join(config["input"]["scenario_dir"], config["input"]["overall_scenario"])
        output_dir = os.path.join(config["output"]["output_dir"], config["input"]["overall_scenario"])
        os.makedirs(output_dir, exist_ok=True)

        """Process non-powerplant emissions (CCS facilities)."""
        ccs_file = os.path.join(csv_input_dir, "facility_easyhard.csv")
        cs_emis = pd.read_csv(ccs_file)
        cs_emis["scc"] = cs_emis["scc"].astype(int)
        cs_emis.rename(columns={"eis": "EIS_ID", "scc": "SCC"}, inplace=True)

        # Check duplicates
        duplicates = cs_emis[cs_emis.duplicated(keep=False)]
        if duplicates.empty:
            print("Great, no duplicate in CCS emissions")
        else:
            print("Duplicates found in CCS emissions:\n", duplicates)

        # Check overlap with powerplant EIS_IDs
        first_scen, first_emis = next(iter(scen_emis_list.items()))
        egrid_file = os.path.join(csv_input_dir, f"pp_{first_scen}.csv")
        egrid = pd.read_csv(egrid_file)
        egrid.rename(columns={"eis": "EIS_ID", "scc": "SCC"}, inplace=True)

        common_ids = cs_emis["EIS_ID"][cs_emis["EIS_ID"].isin(egrid["EIS_ID"])].unique()

        if len(common_ids) == 0:
            print("No duplicated EIS_ID between CCS and eGRID")
            nei_df = unmapped_df.copy()
        else:
            print("Duplicated EIS_ID between CCS and eGRID:", common_ids)
            nei_df = nei_all_pt.copy()

        # Merge NEI + CCS
        nei_with_non_pp = nei_df.merge(
            cs_emis[["EIS_ID", "SCC"]], on=["EIS_ID", "SCC"], how="inner"
        )
        nei_with_non_pp.drop(columns=["height", "diam", "temp", "velocity", "cambium_gea"], inplace=True)
        nei_with_non_pp = nei_with_non_pp.merge(cs_emis, on=["EIS_ID", "SCC"], how="left")

        unique_id_cols = self.find_minimal_unique_identifier_columns(nei_with_non_pp)
        if unique_id_cols:
            print("Columns that uniquely identify rows:", unique_id_cols)
        else:
            print("No combination of columns uniquely identifies rows.")

        print("cs_emis base sum:", cs_emis.filter(like="_tons_base").sum())
        print("cs_emis final sum:", cs_emis.filter(like="_tons_final").sum())

        # Process both base and final
        for is_base in [True, False]:
            nei_all_pt_final = self.mapping_non_powerplant_to_nei(
                nei_with_non_pp, nei_all_pt,  unique_id_cols,   is_base=is_base)

            nei_all_pt_final = nei_all_pt_final.fillna(0)
            mapped_df = nei_all_pt_final[nei_all_pt_final["was_mapped"] == True].copy()
            unmapped_df = nei_all_pt_final[nei_all_pt_final["was_mapped"] != True].copy()

            suffix = "_base" if is_base else ""
            save_filename = os.path.join(output_dir, f"{first_emis}{suffix}.shp")
            if not mapped_df.empty:
                mapped_df.to_file(save_filename, driver="ESRI Shapefile")
                print(f"Saved mapped data to {save_filename}")
            
            if not is_base:
                # Save the rest (unmapped data) as rest_NEI
                if not unmapped_df.empty:
                    rest_filename = os.path.join(output_dir,  f"{first_emis}_rest_NEI.shp")
                    unmapped_df.to_file(rest_filename, driver='ESRI Shapefile')
                    print(f"Saved unmapped NEI data to {rest_filename}")


    # -------------------------
    # Plotting functions
    # -------------------------

    def process_emissions(self, emis_dir_path, emis_name, config, is_powerplant=True):
        """Process emissions for a given case (powerplant or non-powerplant)."""

        emis_dir_path = config['output']['output_dir'] + config["input"]["overall_scenario"]

        overall_scenario = config["input"]["overall_scenario"]

        
        if is_powerplant:
            file_path1 = os.path.join(emis_dir_path, f'{emis_name}_pp_base.shp')
            file_path2 = os.path.join(emis_dir_path, f'{emis_name}_pp.shp')
            df_name = f'powerplants: {emis_name}'
        else:
            if overall_scenario != "Full_USA":
                file_path1 = os.path.join(emis_dir_path, f'current_easyhard_{overall_scenario}_base.shp')
                file_path2 = os.path.join(emis_dir_path, f'current_easyhard_{overall_scenario}.shp')
            else:
                file_path1 = os.path.join(emis_dir_path, 'current_easyhard_base.shp')
                file_path2 = os.path.join(emis_dir_path, 'current_easyhard.shp')
            df_name = 'others_facilities: same in all runs'

        # read and clean
        final_emis1 = gpd.read_file(file_path1).reset_index(drop=True)
        final_emis2 = gpd.read_file(file_path2).reset_index(drop=True)

        print("debugging ", final_emis1[self.NEI_cols].head())
        # sum pollutants
        sum_final1 = final_emis1[self.NEI_cols].sum()
        sum_final2 = final_emis2[self.NEI_cols].sum()

        # prepare comparison df
        df_compare = pd.DataFrame({
            f'{df_name}_base': sum_final1,
            f'{df_name}_final': sum_final2,
            'final-base': sum_final2 - sum_final1
        })
        df_compare['source'] = df_name

        return df_compare

    def plot_emissions_diff(self, final_diff_emis, new_cols, overall_scenario, output_dir):
        """Plot the emissions difference (Final - Base)."""
        final_diff_emis = final_diff_emis[new_cols]

        ax = final_diff_emis.plot(kind="bar", figsize=(10, 6), width=0.8)

        for p in ax.patches:
            value = p.get_height()
            if not pd.isna(value):
                ax.annotate(
                    f"{value:,.0f}",
                    (p.get_x() + p.get_width() / 2, value / 2),
                    ha='center', va='bottom',
                    fontsize=10, rotation=90, xytext=(0, 2), textcoords="offset points"
                )

        plt.title(f"Pollutant Emissions Change (Final - Base) for {overall_scenario}")
        plt.ylabel("Emissions change (tons)")
        plt.xlabel("Pollutants")
        plt.xticks(rotation=0)
        plt.legend(title="")
        plt.tight_layout()

        out_file = os.path.join(output_dir, "Total_Difference.png")
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Plot saved to {out_file}")


    # -------------------------
    # First plotting
    # -------------------------

    def compare_emissions(self, scen_emis_list, config):

        """comparing powerplant and non-powerplant emissions."""

        emis_dir_path = config['output']['output_dir'] + config["input"]["overall_scenario"]

        overall_scenario = config["input"]["overall_scenario"]

        compare_all = []

        # case 1: powerplants
        for emis_name in scen_emis_list.values():
            df_compare = self.process_emissions(emis_dir_path, emis_name, config, is_powerplant=True)
            compare_all.append(df_compare)

        # case 2: non-powerplants
        df_compare = self.process_emissions(emis_dir_path, None, config, is_powerplant=False)
        compare_all.append(df_compare)

        # combine
        df_all = pd.concat(compare_all)

        # setup output dir
        output_dir = os.path.join(
            config['output']['plots_dir'] + config["input"]["overall_scenario"],
            list(scen_emis_list.values())[0])
        os.makedirs(output_dir, exist_ok=True)

        # pivot
        final_diff_emis = df_all.pivot_table(index=df_all.index, columns="source", values="final-base")

        # column order
        new_cols = [f'powerplants: {emis_name}' for emis_name in scen_emis_list.values()]
        new_cols.append('others_facilities: same in all runs')
        print(f"checking plot bar legend names: {new_cols}")

        # plotting
        self.plot_emissions_diff(final_diff_emis, new_cols, overall_scenario, output_dir)

        return df_all


    # -------------------------
    # Second plotting (original emissions comparison)
    # -------------------------

    def compare_with_original(self, scen_emis_list, config):
        """Compare shapefile emissions with original CSV emissions."""

        overall_scenario = config["input"]["overall_scenario"]
        emis_dir_path = config['output']['output_dir'] + config["input"]["overall_scenario"]

        csv_input_dir = os.path.join(config["input"]["scenario_dir"], config["input"]["overall_scenario"])

        pollutant_final_map = {
            'NOx': 'NOx_tons_final',
            'SOx': 'SO2_tons_final',
            'NH3': 'NH3_tons_final',                                                
            'VOC': 'VOC_tons_final',
            'PM2_5': 'PM2.5_tons_final'}
        pollutant_base_map = {
            'NOx': 'NOx_tons_base',
            'SOx': 'SO2_tons_base',
            'NH3': 'NH3_tons_base',
            'VOC': 'VOC_tons_base',
            'PM2_5': 'PM2.5_tons_base'}
            
        for scen_name, emis_name in scen_emis_list.items():

            for is_base_emission in [True, False]:
                if is_base_emission:
                    pollutant_map = pollutant_base_map
                    file_path1 = os.path.join(emis_dir_path, f'{emis_name}_pp_base.shp')
                    file_path2 = os.path.join(emis_dir_path, 'current_easyhard_base.shp')
                    if overall_scenario != 'Full_USA':
                        file_path2 = os.path.join(emis_dir_path, f'current_easyhard_{overall_scenario}_base.shp')
                else:
                    pollutant_map = pollutant_final_map
                    file_path1 = os.path.join(emis_dir_path, f'{emis_name}_pp.shp')
                    file_path2 = os.path.join(emis_dir_path, 'current_easyhard.shp')
                    if overall_scenario != 'Full_USA':
                        file_path2 = os.path.join(emis_dir_path, f'current_easyhard_{overall_scenario}.shp')

                # read shapefiles
                final_emis1 = gpd.read_file(file_path1).reset_index(drop=True)
                final_emis2 = gpd.read_file(file_path2).reset_index(drop=True)

                # read original CSV
                original_file1 = os.path.join(csv_input_dir, f'pp_{scen_name}.csv')
                original_emis1 = pd.read_csv(original_file1)
                original_emis1 = self.reformat_powerplant(original_emis1)

                original_file2 = os.path.join(csv_input_dir, 'facility_easyhard.csv')
                original_emis2 = pd.read_csv(original_file2)

                # clean
                original_emis2['scc'] = original_emis2['scc'].astype(int)
                original_emis2.rename(columns={'eis_id': 'EIS_ID', 'scc': 'SCC'}, inplace=True)

                # sum
                sum_final1 = final_emis1[self.NEI_cols].sum()
                sum_final2 = final_emis2[self.NEI_cols].sum()
                sum_orig1 = original_emis1[list(pollutant_map.values())].sum()
                sum_orig2 = original_emis2[list(pollutant_map.values())].sum()

                sum_orig1.index = self.NEI_cols
                sum_orig2.index = self.NEI_cols

                suffix = "_base" if is_base_emission else "_final"

                df_compare = pd.DataFrame({
                    f'pp{suffix}': sum_final1,
                    'original_pp': sum_orig1,
                    f'non_pp{suffix}': sum_final2,
                    'original_non_pp': sum_orig2
                })
                
                print(df_compare)

                # plot
                ax = df_compare.plot(kind='bar', figsize=(10, 6), width=0.8)
                for p in ax.patches:
                    value = p.get_height()
                    if not pd.isna(value):
                        ax.annotate(
                            f"{value:,.0f}",
                            (p.get_x() + p.get_width() / 2, value),
                            ha='center', va='bottom',
                            fontsize=8, rotation=90, xytext=(0, 2), textcoords="offset points"
                        )

                plt.title(f"Pollutant Sum Comparison for {emis_name}{suffix}")
                plt.ylabel("Emissions (tons)")
                plt.xlabel("Pollutants")
                plt.xticks(rotation=0)
                plt.legend(title="Dataset")
                plt.tight_layout()

                # setup output dir
                output_dir = os.path.join(
                    config['output']['plots_dir'] + config["input"]["overall_scenario"],
                    list(scen_emis_list.values())[0]
                )
                os.makedirs(output_dir, exist_ok=True)

                out_file = os.path.join(output_dir, f"{emis_name}_{suffix}_emission_diagnostics_plot.png")
                plt.savefig(out_file, dpi=300, bbox_inches='tight')
                plt.close()

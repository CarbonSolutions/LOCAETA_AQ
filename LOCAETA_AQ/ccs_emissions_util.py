import os
import sys
import pandas as pd
import numpy as np
import warnings
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import CRS
import logging
# Suppress all warnings in jupyter notebook
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CCSEmissionProcessor:
    """
    A class to handle CCS emission data processing and NEI integration.
    """
    
    def __init__(self, config):
        """
        Initialize the processor with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with file paths and parameters
        """
        self.config = config
        self.NEI_cols = ['VOC', 'NOx', 'NH3', 'SOx', 'PM2_5']
        self.CCS_subpart_cols = ['VOC_subpart_tons', 'NOX_subpart_tons', 'NH3_subpart_tons',
                                'SO2_subpart_tons', 'PM25_subpart_tons']
        self.CCS_changes_cols = ['VOC_increase_SCC_tons', 'NOX_reduction_subpart_tons', 
                                'NH3_increase_SCC_tons', 'SO2_reduction_subpart_tons', 
                                'PM25_reduction_subpart_tons']
        self.sum_cols = ['NH3_increase_SCC_tons', 'VOC_increase_SCC_tons']
        self.all_species = ['VOC', 'NH3', 'NOX', 'SO2', 'PM25']


    
    def load_and_clean_ccs_data(self, ccs_file_path):
        """
        Load and clean CCS data by removing invalid rows.
        
        Args:
            ccs_file_path (str): Path to the CCS data file
            
        Returns:
            pd.DataFrame: Cleaned CCS data
        """
        logger.info("Loading and cleaning CCS data...")
        cs_emis = pd.read_csv(ccs_file_path, index_col=False)
        
        logger.info(f"Original CCS data shape: {cs_emis.shape}")
        
        # Remove rows with epa_subpart = -1
        indices_to_drop = cs_emis[cs_emis['epa_subpart'] == '-1'].index
        cs_emis.drop(indices_to_drop, inplace=True)
        logger.info(f"After dropping epa_subpart = -1: {cs_emis.shape}")
        
        # Remove rows with missing SCC
        indices_to_drop = cs_emis[cs_emis['scc'].isna()].index
        cs_emis.drop(indices_to_drop, inplace=True)
        logger.info(f"After dropping missing SCC: {cs_emis.shape}")
        
        return cs_emis

    def handle_duplicates(self, cs_emis):
        """
        Handle duplicate rows by analyzing different duplicate cases and summing NH3 and VOC emissions 
        while keeping other values consistent.
        
        Args:
            cs_emis (pd.DataFrame): CCS emission data
            
        Returns:
            pd.DataFrame: Deduplicated CCS data
        """
        logger.info("Handling duplicates...")
        
        # Ensure SCC column is integer and rename columns
        cs_emis['scc'] = cs_emis['scc'].astype(int)
        cs_emis.rename(columns={'eis_id': 'EIS_ID', 'scc': 'SCC'}, inplace=True)
        
        # Find all duplicate rows, including the first occurrence
        all_duplicates = cs_emis[cs_emis.duplicated(keep=False)]
        logger.info("All duplicate rows:")
        logger.info(all_duplicates)
        
        # Identify duplicates
        duplicate_keys = (
            cs_emis.groupby(['EIS_ID', 'SCC'])
            .size()
            .reset_index(name='count')
            .query('count > 1')[['EIS_ID', 'SCC']]
        )
        duplicates = cs_emis.merge(duplicate_keys, on=['EIS_ID', 'SCC'], how='inner')
        duplicates['row_key'] = duplicates.index  # Track original index
        
        logger.info(f"{duplicates.shape}, {cs_emis.shape}")
        
        # Case 1: Multiple ghgrp_facility_id for the same NEI (EIS_ID + SCC)
        case1_keys = (
            duplicates.groupby(['EIS_ID', 'SCC'])['ghgrp_facility_id']
            .nunique()
            .reset_index(name='ghgrp_faci_count')
            .query('ghgrp_faci_count > 1')[['EIS_ID', 'SCC']]
        )
        case1 = duplicates.merge(case1_keys, on=['EIS_ID', 'SCC'])
        case1_row_keys = set(case1['row_key'])
        
        # Exclude Case 1 rows before doing Case 2
        case_others = duplicates[~duplicates['row_key'].isin(case1_row_keys)]
        
        # Case 2: Multiple 'epa_subpart' for the same NEI (EIS_ID + SCC)
        case2_keys = (
            case_others.groupby(['EIS_ID', 'SCC'])['epa_subpart']
            .nunique()
            .reset_index(name='epa_subpart_count')
            .query('epa_subpart_count > 1')[['EIS_ID', 'SCC']]
        )
        case2 = case_others.merge(case2_keys, on=['EIS_ID', 'SCC'])
        case2_row_keys = set(case2['row_key'])
        
        logger.info(f"{case1.shape}, {case2.shape}")
        logger.info("Duplicates head:")
        logger.info(duplicates.head())
        
        combined_case_row_keys = case1_row_keys.union(case2_row_keys)
        logger.info(f"Total unique rows in Case 1 or Case 2:, {len(combined_case_row_keys)}")
        
        unexplained = duplicates[~duplicates['row_key'].isin(combined_case_row_keys)]
        logger.info(f"Unexplained duplicates:, {unexplained.shape[0]}")
        
        # Define species names
        species_to_sum = ['VOC', 'NH3']
        all_species = ['VOC', 'NH3', 'NOX', 'SO2', 'PM25']
        
        # Manually define sum_cols
        sum_cols = ['NH3_increase_SCC_tons', 'VOC_increase_SCC_tons']
        
        # Dynamically identify all species-related columns, excluding those in sum_cols
        CCS_cols = [
            col for col in cs_emis.columns
            if any(sp in col for sp in all_species) and col not in sum_cols
        ]
        
        # Step 1: Identify duplicate (eis_id, scc) pairs - using original column names for consistency
        dup_keys = (
            cs_emis.groupby(['EIS_ID', 'SCC'])
            .size().reset_index(name='count')
            .query('count > 1')[['EIS_ID', 'SCC']]
        )
        
        # Step 2: Split duplicated and non-duplicated rows
        cs_emis['row_key'] = cs_emis.index
        duplicated_rows = cs_emis.merge(dup_keys, on=['EIS_ID', 'SCC'], how='inner')
        non_duplicated_rows = cs_emis[~cs_emis['row_key'].isin(duplicated_rows['row_key'])]
        
        # Step 3: Deduplicate only needed rows
        grouped = duplicated_rows.groupby(['EIS_ID', 'SCC'])
        
        def values_consistent(group, col):
            vals = group[col].dropna().unique()
            return len(vals) <= 1
        
        exclude_consistency_check = sum_cols + ['NH3_increase_subpart_tons', 'VOC_increase_subpart_tons', 'frac_of_VOC_subpart']
        
        # Compare only the columns where values differ between the two rows
        def get_diff_columns(group):
            if group.shape[0] != 2:
                raise ValueError("Expected group with exactly 2 rows")
            
            # Compare the two rows
            diffs = (group.iloc[0] != group.iloc[1]) & ~(group.iloc[0].isna() & group.iloc[1].isna())
            
            # Return only the differing columns
            return group.loc[:, diffs]
        
        dedup_list = []
        for key, group in grouped:
            ref = group.iloc[0]
            consistent = all(values_consistent(group, col) for col in CCS_cols if col not in exclude_consistency_check)
            if not consistent:
                logger.info(f"❌ Inconsistent values for group: {key}")
                logger.info(get_diff_columns(group[CCS_cols]))
                raise ValueError(f"Inconsistent species values in group: {key}")
            
            summed_row = ref.copy()
            for col in sum_cols:
                summed_row[col] = group[col].sum()
            dedup_list.append(summed_row)
        
        dedup_df = pd.DataFrame(dedup_list)
        
        # Step 4: Final dataframe
        cs_emis_final = pd.concat([non_duplicated_rows, dedup_df], ignore_index=True)
        cs_emis_final.drop(columns='row_key', inplace=True)
        
        # Step 5: Sanity check
        logger.info("\n==== 🔍 CCS_cols Sum Check ====")
        before = cs_emis[CCS_cols].sum()
        after = cs_emis_final[CCS_cols].sum()
        diff = after - before
        
        check_df = pd.DataFrame({
            'Before': before,
            'After': after,
            'Diff': diff
        })
        logger.info(check_df)
        
        # Check row counts
        logger.info(f"\nOriginal rows: {cs_emis.shape[0]}, Final rows: {cs_emis_final.shape[0]}")
        logger.info(f"Duplicates handled: {duplicated_rows.shape[0] - dedup_df.shape[0]}")
        
        # Verification step from original code
        # Step 1: Sum from non-duplicated rows
        non_dup_sum = non_duplicated_rows[CCS_cols].sum()
        
        # Step 2: For duplicated groups, get the first row only
        first_of_duplicates = duplicated_rows.groupby(['EIS_ID', 'SCC'], as_index=False).first()
        first_dup_sum = first_of_duplicates[CCS_cols].sum()
        
        # Step 3: Compare to full final deduplicated dataframe
        final_sum = cs_emis_final[CCS_cols].sum()
        
        # Step 4: Combine and compare
        verify_df = pd.DataFrame({
            'Non-Duplicated': non_dup_sum,
            'First of Duplicates': first_dup_sum,
            'Reconstructed (NonDup + FirstDup)': non_dup_sum + first_dup_sum,
            'Final Deduplicated': final_sum,
            'Diff': final_sum - (non_dup_sum + first_dup_sum)
        })
        
        logger.info("\n=== 🔍 Emission Verification ===")
        logger.info(verify_df)
        
        return cs_emis_final
    
    def load_nei_data(self, nei_file_path):
        """
        Load NEI point source emission data.
        
        Args:
            nei_file_path (str): Path to the NEI shapefile
            
        Returns:
            gpd.GeoDataFrame: NEI emission data
        """
        logger.info("Loading NEI data...")
        gdf = gpd.read_file(nei_file_path)
        return gdf
    
    def verify_emissions_consistency(self, gdf, cs_emis):
        """
        Verify consistency between NEI and CCS subpart emissions.
        
        Args:
            gdf (gpd.GeoDataFrame): NEI data
            cs_emis (pd.DataFrame): CCS data
        """
        logger.info("Verifying emissions consistency...")
        
        key_cols = ['EIS_ID', 'SCC']
        
        # Aggregate NEI values by (EIS_ID, SCC)
        gdf_agg = gdf.groupby(key_cols)[self.NEI_cols].sum().reset_index()
        
        # Merge with cs_emis for comparison
        compare_df = pd.merge(
            gdf_agg,
            cs_emis[key_cols + self.CCS_subpart_cols + self.CCS_changes_cols],
            on=key_cols,
            how='inner'
        )
        
        # Compare each NEI vs CCS column
        for nei_col, ccs_col in zip(self.NEI_cols, self.CCS_subpart_cols):
            nei_vals = compare_df[nei_col]
            ccs_vals = compare_df[ccs_col]
            
            match_mask = (
                np.isclose(nei_vals, ccs_vals, equal_nan=True) |
                ((nei_vals.isna() & (ccs_vals == 0)) | ((nei_vals == 0) & ccs_vals.isna()))
            )
            
            mismatch_mask = ~match_mask
            
            if mismatch_mask.any():
                logger.info(f"\n⚠️ Mismatched values for {nei_col} vs {ccs_col}:")
                logger.info(compare_df.loc[mismatch_mask, key_cols + [nei_col, ccs_col]])
                logger.info(compare_df.loc[mismatch_mask, [nei_col, ccs_col]].sum())

        # Compare the net difference (between NEI and CCS subpart) to the changes of CCS emissions. 
        for chg_col, ccs_col in zip(self.CCS_changes_cols, self.CCS_subpart_cols):
            logger.info(compare_df.loc[mismatch_mask, [chg_col, ccs_col]].sum())

    def merge_ccs_with_nei(self, gdf, cs_emis):
        """
        Merge CCS emission changes with NEI data through allocation.
        
        Args:
            gdf (gpd.GeoDataFrame): NEI data
            cs_emis (pd.DataFrame): CCS data
            
        Returns:
            gpd.GeoDataFrame: Merged data with CCS changes allocated
        """
        logger.info("Merging CCS with NEI data...")
        
        key_cols = ['EIS_ID', 'SCC']
        gdf_copy = gdf.copy()
        
        # Step 1: Identify matched (EIS_ID, SCC) pairs
        matched_keys = pd.merge(
            gdf_copy[key_cols].drop_duplicates(),
            cs_emis[key_cols].drop_duplicates(),
            on=key_cols
        )
        
        # Step 2: Split gdf into matched and unmatched
        gdf_matched = gdf_copy.merge(matched_keys, on=key_cols, how='inner')
        gdf_unmatched = gdf_copy.merge(matched_keys, on=key_cols, how='outer', indicator=True).query('_merge == "left_only"').drop(columns='_merge')
        
        # Step 3: Merge CCS values into matched gdf
        gdf_matched = gdf_matched.merge(cs_emis[key_cols + self.CCS_changes_cols], 
                                      on=key_cols, how='left', suffixes=('', '_ccs'))
        
        # Step 4: Allocation function
        def allocate(group):
            n = len(group)
            for nei_col, ccs_col in zip(self.NEI_cols, self.CCS_changes_cols):
                ccs_val = group[ccs_col].iloc[0]
                
                if pd.isna(ccs_val):
                    group[ccs_col] = np.nan
                    continue
                
                nei_vals = group[nei_col]
                total_nei = nei_vals.sum()
                
                if pd.isna(total_nei):
                    group[ccs_col] = np.nan
                elif total_nei > 0:
                    group[ccs_col] = (nei_vals / total_nei) * ccs_val
                else:
                    group[ccs_col] = ccs_val / n
            return group
        
        logger.info("Allocating CCS changes to matched groups...")
        gdf_matched_allocated = gdf_matched.groupby(key_cols, group_keys=False).apply(allocate)
        
        # Drop helper columns
        gdf_matched_allocated = gdf_matched_allocated.drop(
            columns=[f"{c}_ccs" for c in self.CCS_changes_cols], errors='ignore')
        
        # Step 5: Add CCS columns to unmatched data
        for c in self.CCS_changes_cols:
            if c not in gdf_unmatched.columns:
                gdf_unmatched[c] = np.nan
        
        final = pd.concat([gdf_matched_allocated, gdf_unmatched], ignore_index=True)
        
        # Step 6: Handle unmatched CCS emissions
        final_keys = final[key_cols].drop_duplicates()
        unmatched_cs_emis = cs_emis.merge(final_keys, on=key_cols, how='outer', 
                                        indicator=True).query('_merge == "left_only"').drop(columns='_merge')
        
        if not unmatched_cs_emis.empty:
            logger.info(f"Adding {len(unmatched_cs_emis)} unmatched CCS emissions...")
            
            # Get facility info from final based on EIS_ID
            rest_cols = [col for col in final.columns if col not in self.CCS_changes_cols + ['SCC', 'EIS_ID']]
            unmatched_lookup = final.drop_duplicates('EIS_ID')[['EIS_ID'] + rest_cols].set_index('EIS_ID')
            
            # Merge facility columns by EIS_ID
            unmatched_with_rest = unmatched_cs_emis.merge(unmatched_lookup, on='EIS_ID', how='left')
            
            # Set emissions for unmatched cases
            for col in self.CCS_changes_cols:
                if col in ['NOX_reduction_subpart_tons', 'SO2_reduction_subpart_tons', 'PM25_reduction_subpart_tons']:
                    unmatched_with_rest[col] = 0.0
            
            for col in self.NEI_cols:
                unmatched_with_rest[col] = 0.0
            
            # Keep only columns matching final and append
            final_cols = final.columns
            unmatched_with_rest = unmatched_with_rest[[col for col in final_cols if col in unmatched_with_rest.columns]]
            
            final = pd.concat([final, unmatched_with_rest], ignore_index=True)
        
        # Conservation check
        self._check_conservation(cs_emis, final, "FINAL CONSERVATION CHECK")
        
        return final
    
    def compute_final_emissions(self, final_df):
        """
        Compute final CCS emissions based on NEI emissions and CCS changes.
        
        Args:
            final_df (gpd.GeoDataFrame): Data with NEI and CCS changes
            
        Returns:
            gpd.GeoDataFrame: Data with final CCS emissions computed
        """
        logger.info("Computing final CCS emissions...")
        
        # Save original NEI columns and compute new CCS emissions
        for nei_col, ccs_col in zip(self.NEI_cols, self.CCS_changes_cols):
            final_df[f'{nei_col}_nei'] = final_df[nei_col]
            
            if nei_col in ['NH3', 'VOC']:
                final_df[nei_col] = final_df[f'{nei_col}_nei'] + final_df[ccs_col].fillna(0)
            else:
                final_df[nei_col] = final_df[f'{nei_col}_nei'] - final_df[ccs_col].fillna(0)
        
        # Compute total difference for each pollutant
        NEI_cols_renamed = ['VOC_nei', 'NOx_nei', 'NH3_nei', 'SOx_nei', 'PM2_5_nei']
        diff_dict = {}
        for nei_col, ccs_col in zip(NEI_cols_renamed, self.NEI_cols):
            nei_total = final_df[nei_col].sum()
            ccs_total = final_df[ccs_col].sum()
            diff_dict[ccs_col] = ccs_total - nei_total
        
        logger.info("Emission differences (CCS - NEI):")
        logger.info(diff_dict)
        
        return final_df
    
    def create_state_subset(self, gdf_with_ccs, state_fips_prefix, state_name):
        """
        Create a subset of data for a specific state.
        
        Args:
            gdf_with_ccs (gpd.GeoDataFrame): Full CCS data
            state_fips_prefix (str): FIPS code prefix for the state
            state_name (str): Name of the state for output files
            
        Returns:
            tuple: (state_subset, other_states)
        """
        logger.info(f"Creating {state_name} subset...")
        
        # Identify state rows using FIPS
        is_state = gdf_with_ccs['FIPS'].str.startswith(state_fips_prefix)
        
        # Split into state and other states
        gdf_state = gdf_with_ccs[is_state].copy()
        gdf_other = gdf_with_ccs[~is_state].copy()
        
        # For other states: backup original CCS columns, then replace with NEI columns
        NEI_cols_renamed = ['VOC_nei', 'NOx_nei', 'NH3_nei', 'SOx_nei', 'PM2_5_nei']
        for nei_col, ccs_col in zip(NEI_cols_renamed, self.NEI_cols):
            gdf_other[f"{ccs_col}_ccs"] = gdf_other[ccs_col]
            gdf_other[ccs_col] = gdf_other[nei_col]
        
        return gdf_state, gdf_other
    
    def create_co_ccs_subset(self, gdf_with_ccs, co_ccs_file_path):
        """
        Create CO CCS subset based on specific facility list.
        
        Args:
            gdf_with_ccs (gpd.GeoDataFrame): Full CCS data
            co_ccs_file_path (str): Path to CO CCS facility data
            
        Returns:
            tuple: (co_subset, remaining_data)
        """
        logger.info("Creating CO CCS subset...")
        
        # Load CO CCS old data to get EIS_ID and SCC
        co_ccs_old = pd.read_csv(co_ccs_file_path)
        
        # Clean CO CCS data
        indices_to_drop = co_ccs_old[co_ccs_old['scc'].isna()].index
        co_ccs_old.drop(indices_to_drop, inplace=True)
        co_ccs_old.drop(co_ccs_old.filter(regex='frs').columns, axis=1, inplace=True)
        co_ccs_old.rename(columns={'eis_id': 'EIS_ID', 'scc': 'SCC'}, inplace=True)
        co_ccs_old = co_ccs_old.drop_duplicates()
        
        # Subset based on CO CCS facilities
        key_cols = ['EIS_ID', 'SCC']
        gdf_subset = gdf_with_ccs.merge(co_ccs_old[key_cols], on=key_cols, how='inner')
        gdf_rest = gdf_with_ccs.merge(co_ccs_old[key_cols], on=key_cols, how='outer', 
                                     indicator=True).query('_merge == "left_only"').drop(columns=['_merge'])
        
        # Reset other facilities to NEI emissions
        NEI_cols_renamed = ['VOC_nei', 'NOx_nei', 'NH3_nei', 'SOx_nei', 'PM2_5_nei']
        for nei_col, ccs_col in zip(NEI_cols_renamed, self.NEI_cols):
            gdf_rest[f"{ccs_col}_ccs"] = gdf_rest[ccs_col]
            gdf_rest[ccs_col] = gdf_rest[nei_col]
        
        logger.info(f"CO subset facilities: {len(gdf_subset['EIS_ID'].unique())}")
        
        return gdf_subset, gdf_rest
    
    def reset_voc_nh3_to_nei(self, gdf):
        """
        Reset VOC and NH3 emissions back to NEI levels (remove increases).
        
        Args:
            gdf (gpd.GeoDataFrame): Data with CCS emissions
            
        Returns:
            gpd.GeoDataFrame: Data with VOC and NH3 reset to NEI levels
        """
        logger.info("Resetting VOC and NH3 emissions to NEI levels...")
        
        gdf_copy = gdf.copy()  # copy first
        for sp in ['VOC', 'NH3']:
            gdf_copy [f'{sp}_ccs'] = gdf_copy [sp]
            gdf_copy [sp] = gdf_copy [f'{sp}_nei']
        return gdf_copy 

    def exclude_ccs_facilities(self, gdf_with_ccs, cs_emis):
        """
        Exclude all CCS facilities from the dataset, keeping only non-CCS facilities.
        
        Args:
            gdf_with_ccs (gpd.GeoDataFrame): Full data with CCS
            cs_emis (pd.DataFrame): CCS facilities data
            
        Returns:
            gpd.GeoDataFrame: Data without CCS facilities
        """
        logger.info("Excluding CCS facilities...")
        
        # Merge with indicator to identify CCS facilities
        merged = gdf_with_ccs.merge(cs_emis[['EIS_ID', 'SCC']], 
                                   on=['EIS_ID', 'SCC'], how='left', indicator=True)
        
        # Keep only non-CCS facilities
        gdf_remaining = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)
        
        logger.info(f"Original facilities: {gdf_with_ccs.shape[0]}")
        logger.info(f"Remaining after exclusion: {gdf_remaining.shape[0]}")
        
        return gdf_remaining
    
    def create_visualizations(self, gdf, output_dir, title_prefix=""):
        """
        Create emission difference visualizations.
        
        Args:
            gdf (gpd.GeoDataFrame): Data to visualize
            output_dir (str): Output directory for plots
            title_prefix (str): Prefix for plot titles
        """
        logger.info(f"Creating visualizations for {title_prefix}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        NEI_cols_renamed = ['VOC_nei', 'NOx_nei', 'NH3_nei', 'SOx_nei', 'PM2_5_nei']
        
        # Compute differences
        diff_dict = {}
        for nei_col, ccs_col in zip(NEI_cols_renamed, self.NEI_cols):
            nei_total = gdf[nei_col].sum()
            ccs_total = gdf[ccs_col].sum()
            diff_dict[ccs_col] = ccs_total - nei_total
        
        # Create difference plot
        diff_series = pd.Series(diff_dict)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(diff_series.index, diff_series.values)
        plt.ylabel("Difference in Total Emissions (CCS - NEI) [tons]")
        plt.title(f"{title_prefix}Difference Between NEI and CCS Emissions by Pollutant")
        plt.axhline(0, color='black', linestyle='--')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:,.2f}',
                    ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Total_Difference.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

        # plot only all industrial facilities
        is_ccs = (gdf[NEI_cols_renamed].values != gdf[self.NEI_cols].values).any(axis=1)
        gdf_with_ccs_zeroed = gdf.copy()

        # Step 1: identify the rows where CCS changes emissions
        nei_totals = gdf[NEI_cols_renamed].sum()
        ccs_totals = gdf[self.NEI_cols].sum()
        gdf_with_ccs_zeroed.loc[is_ccs, self.NEI_cols] = 0 
        zero_totals = gdf_with_ccs_zeroed[self.NEI_cols].sum()
        plot_df = pd.DataFrame({"NEI": nei_totals.values, "CCS": ccs_totals.values, "Zero_out_CCS":zero_totals.values}, index = self.NEI_cols)

        # Step 3: Plot
        ax = plot_df.plot(kind='bar', figsize = (10,6))
        plt.ylabel("Total Emissions [tons]")
        plt.title("Total NEI vs CCS vs zero out emissions at all industrial facilities")
        plt.axhline(0, color='black',linestyle = '--')
        plt.xticks(rotation=0)

        # add labels on bars
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                ax.text(
                    bar.get_x()+ bar.get_width()/2, 
                    height/2, 
                    f'{height:,.0f}',
                    rotation=90,
                    ha = 'center',
                    va ='bottom' if height >= 0 else 'top'
                )

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'NEI_vs_CCS_vs_zero_out_emissions_all_industrial_facilities.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Create comparison plots for CCS vs NEI vs Zero-out
        gdf_ccs_only = gdf.loc[is_ccs, NEI_cols_renamed + self.NEI_cols]
        
        nei_totals = gdf_ccs_only[NEI_cols_renamed].sum()
        ccs_totals = gdf_ccs_only[self.NEI_cols].sum()
        #zero_totals = pd.Series({col: 0 for col in self.NEI_cols})
        
        plot_df = pd.DataFrame({
            "NEI": nei_totals.values,
            "CCS": ccs_totals.values,
        #    "Zero_out": zero_totals.values
        }, index=self.NEI_cols)
        
        ax = plot_df.plot(kind='bar', figsize=(10, 6))
        plt.ylabel("Total Emissions [tons]")
        plt.title(f"{title_prefix}NEI vs CCS vs Zero-out Emissions at CCS Facilities")
        plt.axhline(0, color='black', linestyle='--')
        plt.xticks(rotation=0)
        
        # Add labels on bars
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height/2, f'{height:,.0f}',
                       rotation=90, ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'nei_vs_ccs_vs_zeroout_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        


        logger.info(f"Visualizations saved to {output_dir}")
    
    def _check_conservation(self, cs_emis, final_df, check_name):
        """
        Helper method to check conservation of emissions.
        
        Args:
            cs_emis (pd.DataFrame): Original CCS data
            final_df (gpd.GeoDataFrame): Final processed data
            check_name (str): Name of the conservation check
        """
        total_sum_original = cs_emis[self.CCS_changes_cols].sum()
        total_sum_allocated = final_df[self.CCS_changes_cols].sum()
        
        logger.info(f"\n=== {check_name} ===")
        logger.info("Original Total CCS Emissions:")
        logger.info(total_sum_original)
        logger.info("Allocated Total CCS Emissions:")
        logger.info(total_sum_allocated)
        
        rel_diff = abs(total_sum_original - total_sum_allocated) / abs(total_sum_original)
        logger.info("Relative Difference:")
        logger.info(rel_diff)
        
        if any(rel_diff > 0.0001):
            logger.info("⚠️ CONSERVATION WARNING!")
        else:
            logger.info("✅ CONSERVATION PASSED!")

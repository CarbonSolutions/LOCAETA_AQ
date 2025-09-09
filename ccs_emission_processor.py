#!/usr/bin/env python
# coding: utf-8

"""
CCS Emission Processing Script
==============================

Re-process USA CCS data and merge with NEI emissions.

Date: July 17, 2025 (Completed: Aug 1, 2025; Converted into python: Sep 8, 2025)
Author: Yunha Lee

This script processes USA CCS emission data by:

1. Cleaning up the CCS emission data 
    - Removing invalid rows (epa_subpart = -1, scc = NaN)
    - Handling duplicates (see below for the details)
2. Merging the clean CCS data with NEI point source emissions
3. Computing final CCS emissions with changes
    - Create a USA wide CCS scenario
    - Create a single state CCS scenario (e.g., LA_CCS)
    - Create specific facilities CCS scenario (e.g., CO_CCS)

4. Creating various output files and visualizations

========================================
Kelly's USA CCS data has three types of duplicates when sorting by EIS_ID and SCC:
 
Case 1. multiple sources (most commonly two sources; for example, ptnonipm_2 and ptegu_1)
==>  I plan to split the final CCS emissions into two or more parts, weight by the corresponding NEI emissions.

Case 2. multiple ghgrp facilities are linked (usually 2), which has different NH3 and VOC increase emissions; all other species has identical values.
==>  I plan to sum the NH3 and VOC increase emissions across the two (or more) ghgrp facilities.
 
Case 3. Two subparts (C and D), which result in different NH3 and VOC emissions increase (the other emissions are identical).  
==>  I plan to sum the NH3 and VOC increase emissions from both subparts.
========================================

"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import CRS
import yaml
from LOCAETA_AQ.ccs_emissions_util import CCSEmissionProcessor

# Suppress all warnings
warnings.filterwarnings('ignore')

def load_config(yaml_file: str) -> dict:
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """
    Main function to run the complete CCS emission processing workflow.
    """
    # Configuration
    cfg = load_config("config.yaml")
    config = cfg['ccs_emissions']

    # Create output directories if not exists
    os.makedirs(config['output']['output_dir'], exist_ok=True)
    os.makedirs(config['output']['plots_dir'], exist_ok=True)

    # Initialize processor
    processor = CCSEmissionProcessor(config)
    
    # Step 1: Load and clean CCS data
    cs_emis_raw = processor.load_and_clean_ccs_data(config['input']['ccs_raw_file'])
    
    # Step 2: Handle duplicates
    cs_emis_clean = processor.handle_duplicates(cs_emis_raw)
    
    # Add missing species if needed
    CCS_missing_species = ['VOC', 'NH3']
    for missing in CCS_missing_species:
        missing_name = missing + "_out_subpart_tons"
        if missing_name not in cs_emis_clean.columns:
            print(f"Computing missing species output: {missing_name}")
            cs_emis_clean[missing_name] = (cs_emis_clean[missing + '_subpart_tons'].fillna(0) + 
                                         cs_emis_clean[missing + '_increase_SCC_tons'])
    
    # Drop unnecessary columns
    columns_to_drop = [col for col in cs_emis_clean.columns if "CO2" in col or "cost_" in col]
    cs_emis_clean.drop(columns=columns_to_drop, inplace=True)
    
    # Save processed CCS data
    output_file = os.path.join(config['output']['ccs_clean_file'])
    cs_emis_clean.to_csv(output_file, index=False)
    print(f"Saved processed CCS data to {output_file}")
    
    # Step 3: Load NEI data
    gdf = processor.load_nei_data(config['input']['combined_NEI_emis_path'])
    
    # Add missing PM25_reduction_subpart_tons if needed
    if 'PM25_reduction_subpart_tons' not in cs_emis_clean.columns:
        print('PM25_reduction_subpart_tons is missing, computing it now')
        cs_emis_clean['PM25_reduction_subpart_tons'] = (cs_emis_clean['PM25CON_reduction_subpart_tons'] + 
                                                       cs_emis_clean['PM25FIL_reduction_subpart_tons'])
    
    # Step 4: Verify emissions consistency
    processor.verify_emissions_consistency(gdf, cs_emis_clean)
    
    # Step 5: Merge CCS with NEI data
    final_df = processor.merge_ccs_with_nei(gdf, cs_emis_clean)
    
    # Step 6: Compute final CCS emissions
    final_with_ccs = processor.compute_final_emissions(final_df)
    
    # Step 7: Drop CCS change columns and save main output
    final_with_ccs.drop(processor.CCS_changes_cols, axis=1, inplace=True)
    
    # Save whole USA CCS file
    usa_ccs_file = os.path.join(config['output']['output_dir'], 'whole_USA_CCS.shp')
    final_with_ccs.to_file(usa_ccs_file, driver='ESRI Shapefile')
    print(f"Saved USA CCS data to {usa_ccs_file}")
    
    # Step 8: Create visualizations for whole USA
    us_plots_dir = os.path.join(config['output']['plots_dir'], 'USA_CCS')
    processor.create_visualizations(final_with_ccs, us_plots_dir, "USA ")

    # Step 9: Create version without VOC and NH3 increases
    final_no_voc_nh3 = processor.reset_voc_nh3_to_nei(final_with_ccs)
    no_voc_nh3_file = os.path.join(config['output']['output_dir'], 'whole_USA_CCS_wo_NH3_VOC.shp')
    final_no_voc_nh3.to_file(no_voc_nh3_file, driver='ESRI Shapefile')
    print(f"Saved USA CCS without VOC/NH3 increases to {no_voc_nh3_file}")
    

    ########################################################################
    # Special case 1: Create Louisiana (LA) subset (FIPS starts with '22')
    ########################################################################

    for case in config['special_cases']:
        if not case.get('run', False):
            continue  # Skip if not running

        ctype = case['type']

        if ctype == "state_all_facilities":
            gdf_subset, gdf_rest = processor.create_state_subset(final_with_ccs, case['fips'], case['name'])
            gdf_subset.to_file(os.path.join(config['output']['output_dir'], f"{case['name']}_CCS.shp"), driver='ESRI Shapefile')
            gdf_subset_no_voc_nh3 = processor.reset_voc_nh3_to_nei(gdf_subset)
            gdf_subset_no_voc_nh3.to_file(os.path.join(config['output']['output_dir'], f"{case['name']}_CCS_wo_NH3_VOC.shp"), driver='ESRI Shapefile')
            gdf_rest.to_file(os.path.join(config['output']['output_dir'], f"USA_CCS_without_{case['name']}.shp"), driver='ESRI Shapefile')
            processor.create_visualizations(gdf_subset,
                                            os.path.join(config['output']['plots_dir'], f"{case['name']}_CCS"),
                                            case['name'] + " ")

        elif ctype == "state_specific_facilities":
            facilities_file = case['facilities_file']
            gdf_subset, gdf_rest = processor.create_co_ccs_subset(final_with_ccs, facilities_file)
            gdf_subset.to_file(os.path.join(config['output']['output_dir'], f"{case['name']}_CCS.shp"), driver='ESRI Shapefile')
            gdf_subset_no_voc_nh3 = processor.reset_voc_nh3_to_nei(gdf_subset)
            gdf_subset_no_voc_nh3.to_file(os.path.join(config['output']['output_dir'], f"{case['name']}_CCS_wo_NH3_VOC.shp"), driver='ESRI Shapefile')
            gdf_rest.to_file(os.path.join(config['output']['output_dir'], f"USA_CCS_without_{case['name']}_CCS.shp"), driver='ESRI Shapefile')
            processor.create_visualizations(gdf_subset,
                                            os.path.join(config['output']['plots_dir'], f"{case['name']}_CCS"),
                                            case['name'] + " ")

        elif ctype == "industrial_no_ccs":
            gdf_no_ccs = processor.exclude_ccs_facilities(final_with_ccs, cs_emis_clean)
            gdf_no_ccs.to_file(os.path.join(config['output']['output_dir'], 'USA_CCS_without_CCS_facilities.shp'),
                               driver='ESRI Shapefile')
    
    # Last step: Print final summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE - SUMMARY")
    print("="*60)
    
    NEI_cols_renamed = ['VOC_nei', 'NOx_nei', 'NH3_nei', 'SOx_nei', 'PM2_5_nei']
    
    print(f"\nOriginal NEI emissions sum:")
    nei_totals = gdf[processor.NEI_cols].sum()
    for i, col in enumerate(processor.NEI_cols):
        print(f"  {col}: {nei_totals.iloc[i]:,.0f} tons")
    
    print(f"\nFinal CCS emissions sum:")
    ccs_totals = final_with_ccs[processor.NEI_cols].sum()
    for i, col in enumerate(processor.NEI_cols):
        print(f"  {col}: {ccs_totals.iloc[i]:,.0f} tons")
    
    print(f"\nEmission differences (CCS - NEI):")
    for i, col in enumerate(processor.NEI_cols):
        diff = ccs_totals.iloc[i] - nei_totals.iloc[i]
        print(f"  {col}: {diff:,.0f} tons")
    
    print(f"\nVisualization plots created in {config['output']['plots_dir']}")
    print("\nProcessing completed successfully!")


if __name__ == "__main__":
    main()
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
import warnings
from LOCAETA_AQ.ccs_emissions_util import CCSEmissionProcessor
from LOCAETA_AQ.config_utils import load_config
import logging

# logging from run_workflow 
logger = logging.getLogger(__name__)

# Suppress all warnings
warnings.filterwarnings('ignore')


def main(cfg):
    """
    Main function to run the complete CCS emission processing workflow.
    """
    # Configuration
    config = cfg['ccs_emissions']
    # Add combined NEI file from cfg
    config['combined_nei_file'] = cfg['nei_emissions']['output']['combined_pt_source_file']

    # Create output directories if not exists
    os.makedirs(config['output']['output_dir'], exist_ok=True)
    os.makedirs(config['output']['plots_dir'], exist_ok=True)

    # Initialize processor
    processor = CCSEmissionProcessor(config)
    
    # Step 1: Load and clean CCS data
    cs_emis_raw = processor.load_and_clean_ccs_data(config['input']['ccs_raw_file_dir'])
    
    # Step 2: Handle duplicates and missing columns and clean up the unnecessary columns
    cs_emis_clean = processor.handle_duplicates(cs_emis_raw)
    
    # Save processed CCS data
    output_file = os.path.join(config['output']['ccs_clean_file_dir'])
    cs_emis_clean.to_csv(output_file, index=False)
    logger.info(f"Saved processed CCS data to {output_file}")
    
    # Step 3: Load NEI data
    config['combined_nei_file'] = os.path.join(cfg['base_dirs']['nei_output_root'],
                                                cfg['nei_emissions']['output']['combined_pt_source_file'])
    gdf = processor.load_nei_data(config['combined_nei_file'])
    
    # Step 4: Verify emissions consistency
    processor.verify_emissions_consistency(gdf, cs_emis_clean)
    
    # Step 5: Merge CCS with NEI data
    final_with_ccs = processor.merge_ccs_with_nei(gdf, cs_emis_clean)
    
    # Step 6: Drop CCS change columns and save main output
    final_with_ccs.drop(processor.CCS_changes_cols, axis=1, inplace=True)
    
    # Step 7: Save whole USA CCS file
    run_name = 'USA_CCS'
    #processor.save_case_output(final_with_ccs, run_name, config['output']['output_dir'], run_name)

    # Step 8: Create visualizations for whole USA
    processor.create_visualizations(final_with_ccs, os.path.join(config['output']['plots_dir'], run_name), "USA ")

    # Step 9: Create version without VOC and NH3 increases
    final_no_voc_nh3 = processor.reset_voc_nh3_to_nei(final_with_ccs)
    run_name_nv = f"{run_name}_wo_NH3_VOC"
    #processor.save_case_output(final_no_voc_nh3, run_name_nv, config['output']['output_dir'], run_name_nv)

    ########################################################################
    # Special cases starts
    ########################################################################

    for case in config['special_cases']:
        ctype = case['type']

        logger.info(f"PROCESSING {ctype} emissions now")
        if ctype == "industrial_no_ccs":

            run_name = 'USA_CCS_without_CCS_facilities'
            gdf_no_ccs = processor.exclude_ccs_facilities(final_with_ccs, cs_emis_clean)
            processor.save_case_output(gdf_no_ccs, run_name, config['output']['output_dir'], run_name)

        elif ctype in {"state_all_facilities", "state_specific_facilities"}:
            if ctype == "state_all_facilities":
                gdf_subset, gdf_rest = processor.create_state_subset(final_with_ccs, case['fips'], case['name'])
    
            elif ctype == "state_specific_facilities":
                facilities_file = case['facilities_file_dir']
                gdf_subset, gdf_rest = processor.create_state_specific_facilities_subset(final_with_ccs, facilities_file)

            # Save subset and rest
            run_name = f"{case['name']}_CCS"
            processor.save_case_output(gdf_subset, run_name, config['output']['output_dir'], run_name)
            processor.save_case_output(gdf_rest, f"USA_CCS_without_{run_name}", config['output']['output_dir'], run_name)

            # Save subset without NH3/VOC
            run_name_nv = f"{case['name']}_CCS_wo_NH3_VOC"
            gdf_subset_no_voc_nh3 = processor.reset_voc_nh3_to_nei(gdf_subset)
            processor.save_case_output(gdf_subset_no_voc_nh3, run_name_nv, config['output']['output_dir'],run_name_nv)

            # Instead of saving gdf_rest again, symlink it
            src_base = os.path.join(config['output']['output_dir'], run_name, f"USA_CCS_without_{run_name}")
            dst_base = os.path.join(config['output']['output_dir'], run_name_nv, f"USA_CCS_without_{run_name_nv}")
            processor.symlink_shapefile(src_base, dst_base)

            # Plots
            processor.create_visualizations(
                gdf_subset,
                os.path.join(config['output']['plots_dir'], run_name),
                case['name'] + " "
            )
        
    logger.info("PROCESSING COMPLETE")

if __name__ == "__main__":


    import logging
    import yaml
    from datetime import datetime

    # start logger 
    logfile = f"run_ccs_emissions_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,  # or DEBUG
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
           # logging.StreamHandler(),
            logging.FileHandler(logfile, mode="w")
        ]
    )

    cfg = load_config("config.yaml")
    main(cfg)
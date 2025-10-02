import warnings
import geopandas as gpd
import pandas as pd
from LOCAETA_AQ.analyze_benmap_utils import Benmap_Analyzer
from LOCAETA_AQ.run_benmap_utils import Benmap_Processor 
from LOCAETA_AQ.config_utils import load_config
import logging
import os

# logging from run_workflow 
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def main(cfg):

    # Initialize processor
    processor = Benmap_Analyzer(cfg)
    run_benmap_processor = Benmap_Processor(cfg)

    config = cfg["benmap"]
    scenario = cfg['stages']['scenario']
    run_names = cfg['stages']['run_names']

    logger.info(f"Analyzing INMAP outputs for {scenario}: {run_names}")

    # Setup directories
    output_dir, json_output_dir = processor.setup_output_dirs(config)

    # Optional state filtering for mapping
    state_regions = config.get('analyze', {}).get('map_only_these_state_regions') or {}

    # Benmap configuration
    benmap_root = config['input']['batchmode_dir']
    grid_shapefile_path = cfg["benmap"]['default_setup']['shapefile_dir'] 
    grid_level =  config['default_setup']['grid_level'] 
    target_year = config['default_setup']['target_year'] 

    benmap_output_type =['incidence' , 'valuation'] 

    # Collect run info
    output_pairs = run_benmap_processor.build_output_pairs(scenario, run_names, grid_level, target_year)
    logger.info(f"starting {output_pairs}")
        

    # read Benmap grid shapefile
    grid_gdf = gpd.read_file(grid_shapefile_path)
    grid_gdf.rename( columns={"COL":"Col", "ROW": "Row"}, inplace=True)

    # Define regions
    regions = {"Nation": None}
    if state_regions:
        regions.update(state_regions)

    for run_name, paths in output_pairs.items():
        control_run = paths["sens"]
        base_run = paths["base"]

        logger.info(f"Starting BenMAP analysis for: {run_name}")
        logger.info(f"  Base run: {base_run}")
        logger.info(f"  Control run: {control_run}")

        # Create per-run output dirs
        run_output_dir = os.path.join(output_dir, run_name)
        run_json_output_dir = os.path.join(json_output_dir, run_name)
        os.makedirs(run_output_dir, exist_ok=True)
        os.makedirs(run_json_output_dir, exist_ok=True)

        for benmap_output in benmap_output_type:

            benmap_output_file = f'{benmap_root}/APVR/{control_run}-{benmap_output}.csv'

            # STEP 1: Process BenMAP output and merge with grid
            final_df = processor.process_benmap_output(benmap_output_file, grid_gdf, benmap_output)

            # STEP 2: Analyze and plot by region
            for region_name, state_fips in regions.items():
                logger.info(f"Processing region: {region_name}")

                # Subset the final DataFrame based on the chosen state or national
                final_df_subset = processor.subset_data(final_df, state_fips)

                # STEP 2a: Aggregation and Save summary  
                processor.analyze_region(final_df_subset, region_name, benmap_output, run_output_dir)

                # STEP 3: Create spatial plots
                processor.plot_region_maps(final_df_subset, region_name, benmap_output, run_output_dir)

            # STEP 4: Creat JSON output from BenMAP output 
            # Note : Don't call this before STEP 2 (because final_df gets modified during the call)
            processor.save_grouped_benmap_json(final_df, run_json_output_dir, benmap_output)

    logger.info("BenMAP analysis completed successfully.")

if __name__ == "__main__":

    import logging
    import yaml
    from datetime import datetime
    import argparse

    # start logger 
    logfile = f"log_files/analyze_benmap_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,  # or DEBUG
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
           # logging.StreamHandler(),
            logging.FileHandler(logfile, mode="w")
        ]
    )
    
    parser = argparse.ArgumentParser(description="Analyze benmap outputs.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg)

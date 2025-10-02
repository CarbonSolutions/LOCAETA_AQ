import os
import yaml
import pandas as pd
import numpy as np
import warnings
import geopandas as gpd
from LOCAETA_AQ.nei_emissions_utils import NEIEmissionProcessor
from LOCAETA_AQ.config_utils import load_config
import logging

# logging from run_workflow 
logger = logging.getLogger(__name__)

# Suppress all warnings
warnings.filterwarnings('ignore')

def main(cfg):
    """
    Main function to process NEI SMOKE csv files to INMAP-compliant shapefiles.
    """
    # Configuration
    config = cfg['nei_emissions']

    # Create output directories if not exists
    os.makedirs(config['output']['output_dir'], exist_ok=True)
    os.makedirs(config['output']['plots_dir'], exist_ok=True)

    # Initialize processor
    processor = NEIEmissionProcessor(config)

    # directory with the NEI csv files 
    nei_raw_data_dir = config['input']['nei_raw_data_dir']
    
    # directory where the final emission shapefiles will be saved 
    final_emis_dir = config['output']['output_dir']

    # read county polygon information from this shapefile (needed for non-point sources)
    county_shapefile_path = config['input']['county_shapefile_dir']
    gdf_fips = gpd.read_file(county_shapefile_path)
    gdf_fips['FIPS'] = gdf_fips['STATEFP'].astype(str) + gdf_fips['COUNTYFP'].astype(str)

    # get the list of all files in the NEI raw directory
    all_files = processor.list_all_files(nei_raw_data_dir)
    files_dict = processor.get_dict(all_files)
    clean_list = processor.filter_and_delete_keys(files_dict) # Deleting any files containing the specified substrings (excluding certain keys)

    logger.info("Filtered files_dict:")
    for key, files in clean_list.items():
        logger.info(f"{key}:")
        for file in files:
            logger.info(f"  {file}")

    # loop over all files (items) under each subdirectory (key) 
    for subdir, files in clean_list.items():
        
        # annual netcdf onroad and rail emissions are directly available from 
        # zip files (from https://gaftp.epa.gov/air/emismod/2020/2020emissions/)
        if subdir in ['onroad', 'rail']:
            logger.info( "onroad and rail emissions are processed from annual netcdf")
            continue

        logger.info(f"Files in subdir: {subdir}")
        for i, file in enumerate(files):

            # final emission shapefile name
            output_file = os.path.join(final_emis_dir, f"{subdir}_{i+1}.shp")
            if os.path.exists(output_file):
                logger.info(f"File {output_file} already exists, skipping.")
                continue

            final_df, is_point = processor.process_nei_file(file)

            if is_point:
                final_gdf = gpd.GeoDataFrame(final_df, geometry='coords', crs='epsg:4269')
            else:
                if subdir == 'onroad':
                    logger.info("onroad is too big to use polygon-based emissions.")                 
                else: 
                    final_df = pd.merge(final_df, gdf_fips[['FIPS', 'geometry']], on='FIPS', how='left')
                    final_gdf = gpd.GeoDataFrame(final_df, geometry='geometry', crs='epsg:4269')

            # Debugging purpose - update the CSV file with the total emissions for the current file
            processor.save_state_emis(file, final_gdf)

            # convert the emissions into INMAP target_proj
            final_gdf = processor.reproject_and_save_gdf(final_gdf)

            final_gdf.to_file(output_file)
            logger.info(f"{file} is saved as shapefile here: {output_file}")

    # I created combined point source for industrial source to handle decarbonization emissions later
    processor.combined_point_sources()

    # Convert NetCDF to shapefile for onroad and rail
    processor.netcdf_to_shapefile(gdf_fips)

if __name__ == "__main__":

    import logging
    import yaml
    from datetime import datetime
    import argparse

    # start logger 
    logfile = f"log_files/process_nei_emissions_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,  # or DEBUG
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
           # logging.StreamHandler(),
            logging.FileHandler(logfile, mode="w")
        ]
    )
    
    parser = argparse.ArgumentParser(description="Run nei emissions.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg)
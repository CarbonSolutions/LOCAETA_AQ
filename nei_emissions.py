import os
import yaml
import pandas as pd
import numpy as np
import warnings
import geopandas as gpd
from LOCAETA_AQ.nei_emissions_util import NEIEmissionProcessor

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
    config = cfg['nei_emissions']

    # Create output directories if not exists
    os.makedirs(config['output']['output_dir'], exist_ok=True)
    os.makedirs(config['output']['plots_dir'], exist_ok=True)

    # Initialize processor
    processor = NEIEmissionProcessor(config)

    # directory with all each source directories created from the extracting NEI zip files.
    # zip files are obtained from https://gaftp.epa.gov/air/emismod/2020/2020emissions/
    nei_raw_data_dir = config['input']['nei_raw_data_dir']
    
    # directory where the final emission shapefiles will be saved 
    final_emis_dir = config['output']['output_dir']

    # read county polygon information from this shapefile (needed for non-point sources)
    county_shapefile_path = config['input']['county_shapefile_path']
    gdf_fips = gpd.read_file(county_shapefile_path)
    gdf_fips['FIPS'] = gdf_fips['STATEFP'].astype(str) + gdf_fips['COUNTYFP'].astype(str)

    # get the list of all files in the directory
    all_files = processor.list_all_files(nei_raw_data_dir)
    files_dict = processor.get_dict(all_files)
    clean_list = processor.filter_and_delete_keys(files_dict) # Deleting any files containing the specified substrings (excluding certain keys)

    print("Filtered files_dict:")
    for key, files in clean_list.items():
        print(f"{key}:")
        for file in files:
            print(f"  {file}")

    # loop over all files (items) under each subdirectory (key) 
    for subdir, files in clean_list.items():
        
        if subdir in ['onroad', 'rail']:
            # onroad is processed from netcdf
            print( "onroad and rail emissions are processed from annual netcdf")
            # onroad emissions are directly available from gaftp and rail csv somehow returns 1.9x higher emissions.
            continue

        print(f"Files in subdir: {subdir}")
        for i, file in enumerate(files):

            # final emission shapefile name
            output_file = final_emis_dir  + f"{subdir}_{i+1}.shp"
            if os.path.exists(output_file):
                print(f"File {output_file} already exists, skipping.")
                continue

            final_df, is_point = processor.process_nei_file(file)

            # Separate coordinates from other data if point source
            if is_point:
                final_gdf = gpd.GeoDataFrame(final_df, geometry='coords', crs='epsg:4269')
            else:
                if subdir == 'onroad':
                    print("onroad is too big to use polygon-based emissions.")                 
                else: 
                    # I like to generate non-point emission with this polygon shapefile but it makes a larger emission file. 
                    final_df = pd.merge(final_df, gdf_fips[['FIPS', 'geometry']], on='FIPS', how='left')

                    # Convert to a GeoDataFrame if you want to perform spatial operations
                    final_gdf = gpd.GeoDataFrame(final_df, geometry='geometry', crs='epsg:4269')


            # Update the CSV file with the total emissions for the current file
            processor.save_state_emis(file, final_gdf, config)

            # convert the emissions into INMAP target_proj
            final_gdf = processor.reproject_and_save_gdf(final_gdf, config)

            final_gdf.to_file(output_file)
            print(f"{file} is saved as shapefile here: {output_file}")

    processor.combined_point_sources(config)

    # Convert NetCDF to shapefile for onroad 
    processor.netcdf_to_shapefile(gdf_fips,config)

if __name__ == "__main__":
    main()
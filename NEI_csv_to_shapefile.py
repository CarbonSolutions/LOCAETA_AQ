import os
import sys
import pandas as pd
import numpy as np
import warnings
import geopandas as gpd
from pyproj import CRS
from collections import defaultdict

# Suppress all warnings
warnings.filterwarnings('ignore')

# Add the path to the main package directory
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'LOCAETA_AQ'))
if package_path not in sys.path:
    sys.path.append(package_path)

import emission_processing

def main():

    # read county polygon information from this shapefile (needed for non-point sources)
    shapefile_path = "/Users/yunhalee/Documents/LOCAETA/NEI_emissions/NEI_2020_gaftp_Jun2024/emiss_shp2020/Census/cb_2020_us_county_500k.shp"
    gdf_fips = gpd.read_file(shapefile_path)
    gdf_fips['FIPS'] = gdf_fips['STATEFP'].astype(str) + gdf_fips['COUNTYFP'].astype(str)
    print(gdf_fips[['STATEFP', 'COUNTYFP', 'FIPS']].head())

    # directory with all each source directories created from the extracting NEI zip files.
    # zip files are obtained from https://gaftp.epa.gov/air/emismod/2020/2020emissions/
    nei_raw_data_dir = '/Users/yunhalee/Documents/LOCAETA/NEI_emissions/NEI_2020_gaftp_Jun2024/2020ha2_cb6_20k/inputs/'
    
    # directory where the final emission shapefiles will be saved 
    final_emis_dir = '/Users/yunhalee/Documents/LOCAETA/RCM/INMAP/evaldata_v1.6.1/2020_nei_emissions/'

    # output CSV file that has total emissions for each state computed for each emission file (verfying emission processing)
    output_state_emis_csv = final_emis_dir + 'total_emissions_by_state_and_species.csv'

    # get the list of all files in the directory
    all_files = emission_processing.list_all_files(nei_raw_data_dir)
    files_dict = emission_processing.get_dict(all_files)
    clean_list = emission_processing.filter_and_delete_keys(files_dict) # Deleting any files containing the specified substrings (excluding certain keys)

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

            final_df, is_point = emission_processing.process_nei_file(file)

            # Separate coordinates from other data if point source
            if is_point:
                final_gdf = gpd.GeoDataFrame(final_df, geometry='coords', crs='epsg:4269')
            else:
                if subdir == 'onroad':
                    #df_fips = pd.read_csv("/Users/yunhalee/Documents/LOCAETA/NEI_emissions/CenPop2020_Mean_CO.txt", dtype={'STATEFP': str, 'COUNTYFP': str})
                    #df_fips['FIPS'] = df_fips['STATEFP'] + df_fips['COUNTYFP']
                    #df_fips = df_fips[['FIPS', 'LATITUDE', 'LONGITUDE']]
                    # onroad is still too big to use polygon-based emissions. 
                    #df = pd.merge(final_df, df_fips, left_on='FIPS', right_on='FIPS', how='left')
                    #pt_geometry = [Point(xy) for xy in zip(df['LONGITUDE'], df['LATITUDE'])]

                    # delete LONGITUDE and LATITUDE from final_gdf (don't need to be there)
                    #df.drop(['LONGITUDE','LATITUDE'], axis=1, inplace = True)
                    #final_gdf = gpd.GeoDataFrame(df, geometry=pt_geometry, crs='epsg:4269')  # This is the CRS that most census data is in 
                
                else: 
                    # I like to generate non-point emission with this polygon shapefile but it makes a larger emission file. 
                    final_df = pd.merge(final_df, gdf_fips[['FIPS', 'geometry']], on='FIPS', how='left')

                    # Convert to a GeoDataFrame if you want to perform spatial operations
                    final_gdf = gpd.GeoDataFrame(final_df, geometry='geometry', crs='epsg:4269')

            # Initialize a dictionary to store the total emissions by state and species
            total_emissions = defaultdict(lambda: defaultdict(float))

            # Loop through each row in the final_gdf to accumulate emissions by state and species
            for index, row in final_gdf.iterrows():
                state_code = row['FIPS'][:2]  # Extract the state code from the FIPS
                total_emissions[state_code]['VOC'] += row['VOC']
                total_emissions[state_code]['NOx'] += row['NOx']
                total_emissions[state_code]['NH3'] += row['NH3']
                total_emissions[state_code]['SOx'] += row['SOx']
                total_emissions[state_code]['PM2_5'] += row['PM2_5']

            # Update the CSV file with the total emissions for the current file
            emission_processing.save_state_emis(file, total_emissions, output_state_emis_csv)

            # convert the emissions into INMAP target_proj
            target_crs = "+proj=lcc +lat_1=33.000000 +lat_2=45.000000 +lat_0=40.000000 +lon_0=-97.000000 +x_0=0 +y_0=0 +a=6370997.000000 +b=6370997.000000 +to_meter=1"
            final_gdf = emission_processing.reproject_and_save_gdf(final_gdf, target_crs)

            final_gdf.to_file(output_file)
            print(f"{file} is saved as shapefile here: {output_file}")


if __name__ == "__main__":
    main()
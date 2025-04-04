import os
import sys
import pandas as pd
import numpy as np
import warnings
import geopandas as gpd
from pyproj import CRS

# Suppress all warnings
warnings.filterwarnings('ignore')

# Add the path to the main package directory
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'LOCAETA_AQ'))
if package_path not in sys.path:
    sys.path.append(package_path)

import emission_processing

def main():

    # output file path for processed emissions 
    combined_NEI_emis_path = '/Users/yunhalee/Documents/LOCAETA/RCM/INMAP/evaldata_v1.6.1/2020_nei_emissions/combined_NEI2020_pt_oilgas_ptegu_ptnonipm.shp'
    NEI_CCS_emis_file = "/Users/yunhalee/Documents/LOCAETA/CS_emissions/Colorado_CCS_combined_NEI_point_oilgas_ptegu_ptnonimps.shp"
    State_emis_file = "/Users/yunhalee/Documents/LOCAETA/CS_emissions/Colorado_point_CCS.shp"
    State_CCS_emis_file = "/Users/yunhalee/Documents/LOCAETA/CS_emissions/Colorado_point_CCS_reduced_emis.shp"
    output_plots_dir ='/Users/yunhalee/Documents/LOCAETA/LOCAETA_AQ/outputs/'

    # CCS and NEI raw data directory
    CCS_raw_file = '/Users/yunhalee/Documents/LOCAETA/CS_emissions/final_output_1_manual_update_noLandfill.csv'
    nei_raw_data_dir = '/Users/yunhalee/Documents/LOCAETA/NEI_emissions/NEI_2020_gaftp_Jun2024/2020ha2_cb6_20k/inputs/'

    # find which NEI emission file to open 
    all_files = emission_processing.list_all_files(nei_raw_data_dir)
    files_dict = emission_processing.get_dict(all_files)
    clean_list = emission_processing.filter_and_delete_keys(files_dict)

    all_nei_gdf = gpd.GeoDataFrame()
    for subdir, files in clean_list.items():
        if subdir in ['ptegu', 'pt_oilgas', 'ptnonipm']:
            for i, file in enumerate(files):
                nei_df, is_point = emission_processing.process_nei_file(file)
                try:
                    nei_gdf = gpd.GeoDataFrame(nei_df, geometry='coords', crs='epsg:4269')
                    all_nei_gdf = pd.concat([all_nei_gdf, nei_gdf], ignore_index=True)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                    continue

    # convert the emissions into INMAP target_proj
    target_crs = "+proj=lcc +lat_1=33.000000 +lat_2=45.000000 +lat_0=40.000000 +lon_0=-97.000000 +x_0=0 +y_0=0 +a=6370997.000000 +b=6370997.000000 +to_meter=1"
    gdf = emission_processing.reproject_and_save_gdf(all_nei_gdf, target_crs)

    # save the emissions as a shapefile
    gdf.to_file(combined_NEI_emis_path)
    print(f"Shapefile saved to: {combined_NEI_emis_path}")

    # process CCS emissions to merge it with NEI emissions
    cs_emis = emission_processing.load_and_process_ccs_emissions(CCS_raw_file)

    # following step is commented out because Kelly's output has all eis_id. 
    ## cs_emis = emission_processing.fill_missing_eis_id(cs_emis, 'https://gaftp.epa.gov/air/nei/2020/data_summaries/Facility%20Level%20by%20Pollutant.zip')
    ##cs_emis = emission_processing.calculate_emission_rates(cs_emis)

    subset_df = emission_processing.subset_and_validate_emissions(gdf, cs_emis)
    merged_df = emission_processing.merge_to_NEI_emissions(gdf, cs_emis)

    # List of original emission columns
    original_emission_cols = ['VOC', 'NOx', 'NH3', 'SOx', 'PM2_5']
    CCS_cols = ['VOC_out_subpart_tons', 'NOX_out_subpart_tons', 'NH3_out_subpart_tons', 
                'SO2_out_subpart_tons', 'PM25_out_subpart_tons']
    
    # Create a mapping for renaming the columns
    rename_mapping = {col: col + '_old' for col in original_emission_cols}
    rename_mapping.update({col + '_new': col for col in original_emission_cols})

    # Rename the columns
    merged_df.rename(columns=rename_mapping, inplace=True)

    print("National Emissions with CCS contains the following columns", merged_df.head())

    if not isinstance(merged_df, gpd.GeoDataFrame):
        raise TypeError("The object is not a GeoDataFrame")
    else:
        if merged_df.crs != CRS.from_string(target_crs):
            raise ValueError(f"The GeoDataFrame CRS does not match the target CRS: {target_crs}")
        else: 
            merged_df.to_file(NEI_CCS_emis_file)
            print(f"Final shapefile saved to: {NEI_CCS_emis_file}")

    # Save only LA point sources
    if not isinstance(subset_df, gpd.GeoDataFrame):
        raise TypeError("The object is not a GeoDataFrame")
    else:
        if subset_df.crs != CRS.from_string(target_crs):
            raise ValueError(f"The GeoDataFrame CRS does not match the target CRS: {target_crs}")
        else: 
            subset_df.to_file(State_emis_file)
            print(f"Filtered shapefile saved to: {State_emis_file}")

    NEI_cols = ['VOC', 'NOx', 'NH3', 'SOx', 'PM2_5']
    CCS_cols = ['VOC_out_subpart_tons', 'NOX_out_subpart_tons', 'NH3_out_subpart_tons', 
                'SO2_out_subpart_tons', 'PM25_out_subpart_tons']

    # Create a mapping for renaming the columns
    rename_mapping = {col: col + '_old' for col in NEI_cols}
    rename_mapping.update({CCS : NEI for NEI, CCS in zip(NEI_cols, CCS_cols)})

    # Rename the columns
    subset_df.rename(columns=rename_mapping, inplace=True)

    print("LA Emissions with CCS contains the following columns", subset_df.head())

    # compare CCS facility emissions total with NEI facility total emissions (facility emissions should match with NEI)
    emission_processing.plot_CCS_facility_emissions(subset_df, output_plots_dir)

    if not isinstance(subset_df, gpd.GeoDataFrame):
        raise TypeError("The object is not a GeoDataFrame")
    else:
        if subset_df.crs != CRS.from_string(target_crs):
            raise ValueError(f"The GeoDataFrame CRS does not match the target CRS: {target_crs}")
        else: 
            subset_df.to_file(State_CCS_emis_file)
            print(f"Filtered shapefile saved to: {State_CCS_emis_file}")

if __name__ == "__main__":
    main()
import os
import sys
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Add the path to the main package directory
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'LOCAETA_AQ'))
if package_path not in sys.path:
    sys.path.append(package_path)

import inmap_to_benmap

def main():

    # grid shape file path
    #grid_shapefile_path = "/Users/yunhalee/Documents/LOCAETA/NEI_emissions/NEI_2020_gaftp_Jun2024/emiss_shp2020/Census/cb_2020_us_county_500k.shp"
    grid_shapefile_path = '/Users/yunhalee/Documents/LOCAETA/RCM/BenMAP/grids/US Census Tracts/US Census Tracts.shp'
    grid_shapefile_path = '/Users/yunhalee/Documents/LOCAETA/RCM/BenMAP/grids/County/County.shp'
    grid_level =  'county' # or 'tracts' 
    target_year = '2020'

    inmap_output_dir = '/Users/yunhalee/Documents/LOCAETA/RCM/INMAP/inmap-1.9.6-gridsplit/outputs/'

    # List of InMAP output files to process
    inmap_outputs = [
            'base_nei2020/2020nei_output_run_steady.shp' #, (include only if it is not available)
           # 'CO_CCS/2020nei_output_run_steady.shp',
           # 'CO_CCS_wo_NH3_VOC/2020nei_output_run_steady.shp',
           # 'CO_Suncor_CCS_wo_NH3_VOC/2020nei_output_run_steady.shp',
           # 'CO_Cherokee_CCS_wo_NH3_VOC/2020nei_output_run_steady.shp',
           # 'NEI_no_Landfill_2001411/2020nei_output_run_steady.shp'
    ]

    # Directory to save the grid and PM2.5 csv files, which are BenMAP input files
    benmap_input_dir =f'/Users/yunhalee/Documents/LOCAETA/RCM/BenMAP/inmap_output/{grid_level}_case/'

    for inmap_output_path in inmap_outputs:

        inmap_runname = inmap_output_path.split('/')[0]  
        print(f"Processing InMAP output from directory: {inmap_runname}")

        output_shapefile_path = f'{benmap_input_dir}{inmap_runname}_{grid_level}_inmap_grid.shp' # not used 
        
        if "base" in inmap_runname:
            output_csv_path = f'{benmap_input_dir}base_{grid_level}_inmap_{target_year}_pm25.csv'
        else: # control
            output_csv_path = f'{benmap_input_dir}control_{inmap_runname}_{grid_level}_inmap_{target_year}_pm25.csv'
        
        inmap, grid_level_gdf = inmap_to_benmap.process_inmap_to_benmap_inputs(
            inmap_output_dir+inmap_output_path, 
            grid_shapefile_path, 
            output_shapefile_path, 
            output_csv_path, 
            grid_level = grid_level
        )
        
        inmap_to_benmap.plot_pm25_original_and_reshaped_results(inmap, grid_level_gdf, inmap_runname, benmap_input_dir,grid_level)


if __name__ == "__main__":
    main()
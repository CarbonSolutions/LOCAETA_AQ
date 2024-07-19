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

    # County shape file path
    county_shapefile_path = "/Users/yunhalee/Documents/LOCAETA/NEI_emissions/NEI_2020_gaftp_Jun2024/emiss_shp2020/Census/cb_2020_us_county_500k.shp"

    inmap_output_dir = '/Users/yunhalee/Documents/LOCAETA/RCM/INMAP/inmap-1.9.6/outputs/'

    # List of InMAP output files to process
    inmap_outputs = [
    'nei2020/2020nei_output_run_steady.shp',
        'nei2020_LA_CCS/2020nei_output_run_steady.shp',
    ]

    # Directory to save the grid and PM2.5 csv files, which are BenMAP input files
    benmap_input_dir ='/Users/yunhalee/Documents/LOCAETA/RCM/BenMAP/immap_output/'

    for inmap_output_path in inmap_outputs:

        inmap_runname = inmap_output_path.split('/')[0]  
        print(f"Processing InMAP output from directory: {inmap_runname}")

        output_shapefile_path = f'{benmap_input_dir}{inmap_runname}_county_level_inmap_grid.shp'
        output_csv_path = f'{benmap_input_dir}{inmap_runname}_county_level_inmap_PM25.csv'
        
        inmap, county_level_gdf = inmap_to_benmap.process_inmap_to_benmap_inputs(
            inmap_output_dir+inmap_output_path, 
            county_shapefile_path, 
            output_shapefile_path, 
            output_csv_path
        )
        
        inmap_to_benmap.plot_pm25_original_and_reshaped_results(inmap, county_level_gdf, inmap_runname, benmap_input_dir)


if __name__ == "__main__":
    main()
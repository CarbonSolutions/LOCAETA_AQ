import os
import sys
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Add the path to the main package directory
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'LOCAETA_AQ'))
if package_path not in sys.path:
    sys.path.append(package_path)

import model_evaluation

def main():

    # Define base paths for INMAP output shapefiles and AQS files
    inmap_output_path = '/Users/yunhalee/Documents/LOCAETA/RCM/INMAP/inmap-1.9.6/outputs/' 
    aqs_file_path = '/Users/yunhalee/Documents/LOCAETA/RCM/EPA_AQS/' 
    eval_output_dir = '/Users/yunhalee/Documents/LOCAETA/LOCAETA_AQ/outputs/'

    # Define base paths for satellite output netcdf files
    satellite_netcdf_path = '/Users/yunhalee/Documents/LOCAETA/RCM/Satellite-PM2.5/'

    # Runs you want to evaluation 
    inmap_runs = {2005: 'nei2005/2005nei_output_run_steady.shp',
                2014:  'nei2014/2014nei_output_run_steady.shp', 
                2020: 'nei2020/2020nei_output_run_steady.shp'}

    satellites ={2005: 'V5GL02.HybridPM25.NorthAmerica.200501-200512.nc', 
                2014: 'V5GL02.HybridPM25.NorthAmerica.201401-201412.nc', 
                2020: 'V5GL02.HybridPM25.NorthAmerica.202001-202012.nc'} 

    # Threshold and threshold type parameters
    threshold = 0
    threshold_type = 'more'

    inmap_variable = 'TotalPM25'
    satellite_variable = 'GWRPM25'

    # Process INMAP data
    model_evaluation.process_inmap(inmap_runs, inmap_output_path, inmap_variable, aqs_file_path, eval_output_dir, threshold, threshold_type)

    # Process Satellite data
    model_evaluation.process_satellite(satellites, satellite_netcdf_path, satellite_variable, aqs_file_path, eval_output_dir, threshold, threshold_type)


if __name__ == "__main__":
    main()
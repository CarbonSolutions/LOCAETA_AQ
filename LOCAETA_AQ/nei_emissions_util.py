# my_emissions.py


import pandas as pd
import io
import zipfile
import os
from shapely.geometry import Point
import geopandas as gpd
from pyproj import Proj, transform
import matplotlib.pyplot as plt
from pyproj import CRS
import numpy as np
import xarray as xr
from collections import defaultdict

# Suppress all warnings in jupyter notebook
import warnings
warnings.filterwarnings('ignore')


class NEIEmissionProcessor:
    """
    A class to handle NEI emission data processing from NEI-SMOKE formatted csv to shapefiles
    """
    
    def __init__(self, config):
        """
        Initialize the processor with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with file paths and parameters
        """
        self.config = config

    # Function to extract a single zip file
    def extract_zip(self, zip_path, extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                # Get the member's path
                member_path = os.path.normpath(member)
                # Form the absolute path to extract to
                full_path = os.path.join(extract_to, member_path)

                # Check if it's a directory
                if member.endswith('/'):
                    if not os.path.exists(full_path):
                        os.makedirs(full_path)
                else:
                    # Ensure the directory exists for the file
                    dir_name = os.path.dirname(full_path)
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    # Extract the file
                    with zip_ref.open(member) as source, open(full_path, "wb") as target:
                        target.write(source.read())


    # Function to list all files in a directory and return their full paths grouped by subdirectory
    def list_all_files(self, directory):
        files_by_subdir = {}
        for root, dirs, files in os.walk(directory):
            subdir = os.path.relpath(root, directory)
            subdir_files = []
            for file in files:
                full_path = os.path.join(root, file)
                if "hourly" not in full_path and "monthly" not in full_path and "nexthour" not in full_path and full_path.endswith('.csv'):
                    subdir_files.append(full_path)
            if subdir_files:
                files_by_subdir[subdir] = subdir_files
        return files_by_subdir

    def get_dict(self, all_files):
        # Generate the output in dictionary format
        files_dict = {}
        for subdir, files in all_files.items():
            if subdir == ".":
                subdir = "root"
            files_dict[subdir] = files
        return files_dict

    # Function to filter and delete keys
    def filter_and_delete_keys(self, files_dict):
        # List of substrings to check in file paths
        substrings_to_check = ["_MX_", "_CA_", "canada", "mexico", "SPEED", "VPOP", "_haps_", "nonCONUS"]
        
        # List of keys to skip
        keys_to_skip = ["onroad_ca_adj", "canmex_ag", "canmex_point"]

        # Delete the marked keys
        for key in keys_to_skip:
            if key in files_dict:
                del files_dict[key]
                print(f"Deleted key: {key}")

        # Iterate over each key in the dictionary
        for key in list(files_dict.keys()):
            if key in keys_to_skip:
                print(f"Skipping key: {key}")
                continue  # Skip the specified keys
            
            print(f"Processing key: {key}")
            
            files_to_keep = []
            for file in files_dict[key]:
                if not any(substring in file for substring in substrings_to_check):
                    files_to_keep.append(file)
                else:
                    print(f"Excluding file due to match: {file}")

            files_dict[key] = files_to_keep
        return files_dict

    # Function to safely convert a value to float and round it 
    # decimals doesn't seem working properly.
    def safe_float_conversion(self, value, conversion_factor=1, offset=0, nan_value=0, decimals=5):
        try:
            if pd.isnull(value) or value == '':
                return nan_value
            # Perform the conversion and rounding
            return round(float(value) * conversion_factor + offset, decimals)
        except ValueError:
            return nan_value
        
    def reproject_and_save_gdf(self, gdf, config):

        # Check if the GeoDataFrame is already in the target CRS
        current_crs = gdf.crs
        target_proj = CRS.from_string(config['input']['target_crs'])
        
        if current_crs != target_proj:
            print(f"Reprojecting from {current_crs} to {target_proj}")
            gdf = gdf.to_crs(config['input']['target_crs'])
        else:
            print(f"GeoDataFrame is already in the target CRS: {target_proj}")
        return gdf

    # Function to process the emissions data efficiently using pandas
    def process_emissions_data(self, emis_df, is_point):
        # Initialize new columns for each pollutant
        emis_df['VOC'] = 0.0
        emis_df['NOx'] = 0.0
        emis_df['NH3'] = 0.0
        emis_df['SOx'] = 0.0
        emis_df['PM2_5'] = 0.0

        # Map pollutants to their respective columns
        poll_mapping = {
            'VOC': 'VOC',
            'VOC_INV': 'VOC',
            'PM25-PRI': 'PM2_5',
            'PM2_5': 'PM2_5',
            'DIESEL-PM25': 'PM2_5',
            'PM25TOTAL': 'PM2_5',
            'NOX': 'NOx',
            'HONO': 'NOx',
            'NO': 'NOx',
            'NO2': 'NOx',
            'NH3': 'NH3',
            'SO2': 'SOx'
        }

        # Apply the mapping to fill the new columns
        for poll, new_col in poll_mapping.items():
            mask = emis_df['poll'] == poll
            emis_df.loc[mask, new_col] = emis_df.loc[mask, 'ann_value']

        # Remove the 'poll' and 'ann_value' columns
        emis_df = emis_df.drop(columns=['poll', 'ann_value'])


        # Convert stack values and add coords for point sources
        if is_point:
            emis_df['height'] = emis_df['stkhgt'].apply(self.safe_float_conversion, args=(0.3048, 0, 0, 5))
            emis_df['diam'] = emis_df['stkdiam'].apply(self.safe_float_conversion, args=(0.3048, 0, 0, 5))
            emis_df['temp'] = emis_df['stktemp'].apply(self.safe_float_conversion, args=(5.0/9.0, 0, 273.15 - (32 * 5.0/9.0), 5))
            emis_df['velocity'] = emis_df['stkvel'].apply(self.safe_float_conversion, args=(0.3048, 0, 0, 5))
            emis_df['coords'] = emis_df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)

        if is_point: 
            stack_cols = ['height', 'diam', 'temp', 'velocity']
            group_all = ['facility_id', 'rel_point_id', 'scc']
            group_short = ['facility_id', 'scc']

            # First, ensure each (facility_id, rel_point_id, scc) has all species in one row
            # Aggregation functions
            agg_funcs_indivual = {
                'region_cd': 'first',
                #'scc': 'first',
                'VOC': 'sum',
                'NOx': 'sum',
                'NH3': 'sum',
                'SOx': 'sum',
                'PM2_5': 'sum',
                'height': 'first',
                'diam': 'first',
                'temp': 'first',
                'velocity': 'first',
                #'facility_id': 'first',
                'coords': 'first',
                'latitude': 'first',
                'longitude': 'first'
            }
            
            # Aggregate by (facility_id, rel_point_id, scc) to get all species in one row
            emis_df = emis_df.groupby(group_all).agg(agg_funcs_indivual).reset_index()
            
            # Now proceed with stack parameter checking
            emis_df['stack_hash'] = emis_df[stack_cols].astype(str).agg('-'.join, axis=1)
            
            # Count both unique stack hashes AND unique rel_point_ids per (facility_id, scc)
            group_stats = emis_df.groupby(group_short).agg({
                'stack_hash': 'nunique',
                'rel_point_id': 'nunique'
            }).rename(columns={'stack_hash': 'n_stack_variants', 'rel_point_id': 'n_rel_points'})
            
            # Only aggregate if there are multiple rel_point_ids with identical stack parameters
            to_aggregate_keys = group_stats[
                (group_stats['n_stack_variants'] == 1) & 
                (group_stats['n_rel_points'] > 1)
            ].index
            
            # Flag rows that can be aggregated
            emis_df['can_aggregate'] = emis_df.set_index(group_short).index.isin(to_aggregate_keys)
            
            df_agg = emis_df[emis_df['can_aggregate']].copy()
            df_rest = emis_df[~emis_df['can_aggregate']].copy()

            # Aggregation functions
            agg_funcs = {
                'region_cd': 'first',
                'scc': 'first',
                'VOC': 'sum',
                'NOx': 'sum',
                'NH3': 'sum',
                'SOx': 'sum',
                'PM2_5': 'sum',
                'height': 'first',
                'diam': 'first',
                'temp': 'first',
                'velocity': 'first',
                'facility_id': 'first',
                'coords': 'first',
                'latitude': 'first',
                'longitude': 'first'
            }

            # Aggregate those with identical stack info across rel_point_id
            df_agg_result = df_agg.groupby(group_short).agg(agg_funcs).reset_index(drop=True)
            df_agg_result['rel_point_id'] = 'AGG'  # Optional placeholder

            # Keep df_rest as-is (individual rel_point_id rows, but each row has all species)
            df_rest_result = df_rest.copy()

            grouped_emis_df = pd.concat([df_agg_result, df_rest_result], ignore_index=True)

            # Keep the rest as-is
            grouped_emis_df.rename(columns={
                'region_cd': 'FIPS',
                'facility_id': 'EIS_ID',
                'scc': 'SCC'
            }, inplace=True)

            #remove unnecessary columns
            grouped_emis_df.drop(columns=['stack_hash', 'can_aggregate'], inplace=True)
            print(grouped_emis_df.head())

        else:
            # Non-point source logic unchanged
            group_keys = ['region_cd', 'scc']
            aggregation_functions = {
                'region_cd': 'first',
                'scc': 'first',
                'VOC': 'sum',
                'NOx': 'sum',
                'NH3': 'sum',
                'SOx': 'sum',
                'PM2_5': 'sum'
            }
            grouped_emis_df = emis_df.groupby(group_keys).agg(aggregation_functions).reset_index(drop=True)
            grouped_emis_df.rename(columns={'region_cd': 'FIPS', 'scc': 'SCC'}, inplace=True)

        print("grouped_emis_df dataframe")
        print(grouped_emis_df.head())

        return grouped_emis_df

    # Function to update the CSV file with new data
    def save_state_emis(self, file_name, gdf, config):

        # Initialize a dictionary to store the total emissions by state and species
        total_emissions = defaultdict(lambda: defaultdict(float))

        # Loop through each row in the final_gdf to accumulate emissions by state and species
        for index, row in gdf.iterrows():
            state_code = row['FIPS'][:2]  # Extract the state code from the FIPS
            total_emissions[state_code]['VOC'] += row['VOC']
            total_emissions[state_code]['NOx'] += row['NOx']
            total_emissions[state_code]['NH3'] += row['NH3']
            total_emissions[state_code]['SOx'] += row['SOx']
            total_emissions[state_code]['PM2_5'] += row['PM2_5']

        # Convert the dictionary to a DataFrame
        total_emissions_df = pd.DataFrame(total_emissions).T
        total_emissions_df.reset_index(inplace=True)
        total_emissions_df.columns = ['State', 'VOC', 'NOx', 'NH3', 'SOx', 'PM2_5']
        total_emissions_df['File'] = file_name

        # output CSV file that has total emissions for each state computed for each emission file (verfying emission processing)
        output_state_emis_csv = config['output']['output_dir'] +  config['output']['state_emissions_sum']

        # If the output CSV already exists, append to it
        if os.path.exists(output_state_emis_csv):
            existing_df = pd.read_csv(output_state_emis_csv)
            updated_df = pd.concat([existing_df, total_emissions_df], ignore_index=True)
        else:
            updated_df = total_emissions_df

        # Save the updated DataFrame to the CSV file
        updated_df.to_csv(output_state_emis_csv, index=False)

    # This is the main function to read and process a single NEI file
    def process_nei_file(self, file_path):

        emis_df = pd.read_csv(file_path, comment='#', dtype={'region_cd': str})

        print(file_path, emis_df.columns)

        # Determine the type of emissions (point vs nonpoint) based on latitude/longitude columns existence
        is_point = set(['latitude', 'longitude']).issubset(emis_df.columns)

        if is_point :
            # Subset the dataframe to the columns we need
            emis_df = emis_df[['region_cd', 'scc', 'poll', 'ann_value', 
                            'stkhgt', 'stkdiam', 'stktemp', 'stkvel',
                            'facility_id', 'rel_point_id',  #  'process_id', 'unit_id',
                            'latitude', 'longitude']]
        else:
            # Subset the dataframe to the columns we need
            emis_df = emis_df[['region_cd', 'scc', 'poll', 'ann_value']]

        print(emis_df.head())

        # debugging purpose
        #updated_filtered_emis_df = update_filtered_emis_df(emis_df)

        processed_df = self.process_emissions_data(emis_df, is_point)
        # debugging purpose
        #updated_filtered_nei_df = update_filtered_nei_df(processed_df)

        return processed_df, is_point


    def combined_point_sources(self, config):

        # List of shapefiles
        files = ['ptegu_1.shp', 'ptegu_2.shp', 'ptnonipm_1.shp', 'ptnonipm_2.shp', 'pt_oilgas_1.shp' ]

        # Initialize an empty list to collect GeoDataFrames
        gdf_list = []

        for f in files:
            gdf = gpd.read_file(config['output']['output_dir']+ f)
            
            # Add a column with the file name (without '.shp')
            gdf['source_file'] = os.path.splitext(os.path.basename(f))[0]
            
            gdf_list.append(gdf)

        # Concatenate all GeoDataFrames
        gdf_combined = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)

        # Save to new shapefile
        gdf_combined.to_file(config['output']['output_dir']+ config['output']['combined_pt_source_file'])

        print(f"Shapefile saved as {config['output']['combined_pt_source_file']}")


    def netcdf_to_shapefile(self, gdf_fips, config):

        from pathlib import Path
        # Define the input NetCDF file and the output shapefile path
        netcdf_path = Path(config['input']['netcdf_path'])
        output_path = config['output']['output_dir']

        netcdf_files = []
        # get the list of netcdf files to process
        for f in netcdf_path.iterdir():
            if f.is_file() and f.suffix == ".ncf":
                if "cb6_20k_onroad" in f.name or "rail" in f.name:
                    netcdf_files.append(f)  # .stem removes ".ncf"


        for netcdf_file in netcdf_files:

            print("processing ", netcdf_file)
            # Open the NetCDF file using xarray
            ds = xr.open_dataset(netcdf_file)

            # Extract the projection information from attributes
            projection_attrs = ds.attrs

            lat_1 = projection_attrs.get('P_ALP')
            lat_2 = projection_attrs.get('P_BET')
            lat_0 = projection_attrs.get('YCENT')
            lon_0 = projection_attrs.get('XCENT')
            earth_radius = 6370000.0

            # Extract the origin and cell size from global attributes
            x_orig = ds.attrs['XORIG'] 
            y_orig = ds.attrs['YORIG'] 
            x_cell = ds.attrs['XCELL']
            y_cell = ds.attrs['YCELL'] 

            # Create x and y coordinates based on origin and cell size
            cols = ds.dims['COL']
            rows = ds.dims['ROW']
            x = x_orig + np.arange(cols) * x_cell
            y = y_orig + np.arange(rows) * y_cell

            # Create meshgrid for coordinates
            xv, yv = np.meshgrid(x, y)

            # Flatten the coordinate grids
            xv_flat = xv.flatten()
            yv_flat = yv.flatten()

            # Define the projection using pyproj
            lcc_proj = Proj(proj='lcc', lat_1=lat_1, lat_2=lat_2, lat_0=lat_0, lon_0=lon_0, ellps='WGS84')

            # Transform the coordinates to lat/lon
            lon, lat = transform(lcc_proj, Proj(proj='latlong', ellps='WGS84'), xv_flat, yv_flat)

            # Aggregate species into categories
            data_dict = {
                'VOC': np.zeros_like(xv_flat),
                'PM2_5': np.zeros_like(xv_flat),
                'NOx': np.zeros_like(xv_flat),
                'NH3': np.zeros_like(xv_flat),
                'SOx': np.zeros_like(xv_flat)
            }

            for variable_name in ds.data_vars:
                data = ds[variable_name].values.flatten()
                if variable_name in ['VOC', 'VOC_INV', 'XYL', 'TOL', 'TERP', 'PAR', 'OLE', 'NVOL', 'MEOH', 
                                    'ISOP', 'IOLE', 'FORM', 'ETOH', 'ETHA', 'ETH', 'ALD2', 'ALDX', 'CB05_ALD2', 
                                    'CB05_ALDX', 'CB05_BENZENE', 'CB05_ETH', 'CB05_ETHA', 'CB05_ETOH', 
                                    'CB05_FORM', 'CB05_IOLE', 'CB05_ISOP', 'CB05_MEOH', 'CB05_OLE', 'CB05_PAR', 
                                    'CB05_TERP', 'CB05_TOL', 'CB05_XYL', 'ETHANOL', 'NHTOG', 'NMOG', 'VOC_INV']:
                    data_dict['VOC'] += data
                elif variable_name in ['PM25-PRI', 'PM2_5', 'DIESEL_PMFINE'] :
                                    #'DIESEL-PMNO3','DIESEL-PMC', 'DIESEL-PMEC',
                                    #'DIESEL-PMOC', 'DIESEL-PMSO4','PAL', 'PCA', 'PCL', 'PEC', 'PFE', 'PK', 
                                    #'PMG', 'PMN', 'PMOTHR', 'PNH4', 'PNO3', 'POC', 'PSI', 'PSO4', 'PTI']:
                    print("checking variable for PM2_5", variable_name, data)
                    data_dict['PM2_5'] += data
                elif variable_name in ['NOX', 'HONO', 'NO', 'NO2']:
                    data_dict['NOx'] += data
                elif variable_name == 'NH3':
                    data_dict['NH3'] += data
                elif variable_name == 'SO2':
                    data_dict['SOx'] += data

            # Filter out rows where all key emissions are zero
            mask = (data_dict['VOC'] != 0) | (data_dict['PM2_5'] != 0) | (data_dict['NOx'] != 0) | (data_dict['NH3'] != 0) | (data_dict['SOx'] != 0)
            for key in data_dict:
                data_dict[key] = data_dict[key][mask]
            lon = lon[mask]
            lat = lat[mask]

            # Create a GeoDataFrame
            geometry = [Point(lon[i], lat[i]) for i in range(len(lon))]
            gdf = gpd.GeoDataFrame(data_dict, geometry=geometry, crs='EPSG:4326')

            gdf_fips = gdf_fips[['FIPS', 'geometry']]

            # Spatial join to assign points to counties
            gdf_joined = gdf_fips.sjoin(gdf, how='inner', op='intersects')

            # Aggregate emissions by county
            aggregated = gdf_joined.dissolve(by='FIPS', aggfunc='sum')

            # Define the target projection
            target_proj = Proj(config['input']['target_crs'])

            # Reproject the shapefile
            aggregated = aggregated.to_crs(target_proj.srs)

            col_sums_list = ['PM2_5', 'VOC']
            column_sums = aggregated[col_sums_list].sum()
            print("Column Sums:\n", column_sums)

            # Save to shapefile
            output_shapefile = os.path.join(output_path, f"{netcdf_file.stem}.shp")
            aggregated.to_file(output_shapefile, driver='ESRI Shapefile')
            print(f'Shapefile saved to {output_shapefile}')




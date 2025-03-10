# my_emissions.py


import pandas as pd
import io
import zipfile
import requests
import os
import csv
from shapely.geometry import Point
import geopandas as gpd
from pyproj import Proj, transform
import matplotlib.pyplot as plt
from pyproj import CRS
import numpy as np
from collections import defaultdict

# Suppress all warnings in jupyter notebook
import warnings
warnings.filterwarnings('ignore')

# Function to extract a single zip file
def extract_zip(zip_path, extract_to):
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
def list_all_files(directory):
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

def get_dict(all_files):
    # Generate the output in dictionary format
    files_dict = {}
    for subdir, files in all_files.items():
        if subdir == ".":
            subdir = "root"
        files_dict[subdir] = files
    return files_dict

# Function to filter and delete keys
def filter_and_delete_keys(files_dict):
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
def safe_float_conversion(value, conversion_factor=1, offset=0, nan_value=0, decimals=5):
    try:
        if pd.isnull(value) or value == '':
            return nan_value
        # Perform the conversion and rounding
        return round(float(value) * conversion_factor + offset, decimals)
    except ValueError:
        return nan_value

# Don't use add_record_emissions. this is too slow
def add_record_emission(row, column_indices, emissions_data, is_point):

    if is_point:
        pol_idx, emis_idx, fips_idx, scc_idx, hgt_idx, diam_idx, temp_idx, vel_idx, lat_idx, lon_idx = column_indices
    else:
        pol_idx, emis_idx, fips_idx, scc_idx = column_indices

    """ Process one row of the emissions file """
    pol = row[pol_idx]
    emis = row[emis_idx]
    if emis == '': return
    
    # Get FIPS and SCC
    fips = str(row[fips_idx])
    scc = str(row[scc_idx])
    key = (fips, scc)

    if key not in emissions_data:
        emissions_data[key] = {
            'FIPS': fips,
            'SCC': scc,
            'VOC': 0.0, 'NOx': 0.0, 'NH3': 0.0, 'SOx': 0.0, 'PM2_5': 0.0,
        }
        if is_point:
            emissions_data[key].update({
                'height': safe_float_conversion(row[hgt_idx], 0.3048, nan_value=0, decimals=5),
                'diam': safe_float_conversion(row[diam_idx], 0.3048, nan_value=0, decimals=5),
                'temp': safe_float_conversion(row[temp_idx], 5.0/9.0, offset=273.15 - (32 * 5.0/9.0), nan_value=0, decimals=5),
                'velocity': safe_float_conversion(row[vel_idx], 0.3048, nan_value=0, decimals=5),
                'coords': Point(row[lon_idx], row[lat_idx])
            })

    if pol in ['VOC', 'VOC_INV']: # TODO - Far from the complete VOC list, so this must be improved 
        emissions_data[key]['VOC'] += float(emis)
    elif pol in ['PM25-PRI', 'PM2_5', 'DIESEL-PM25', 'EC', 'NH4', 'NO3', 'OC', 'SO4','PM25TOTAL']:
        emissions_data[key]['PM2_5'] += float(emis)
    elif pol in ['NOX', 'HONO', 'NO', 'NO2']:
        emissions_data[key]['NOx'] += float(emis)
    elif pol == 'NH3':
        emissions_data[key]['NH3'] += float(emis)
    elif pol == 'SO2':
        emissions_data[key]['SOx'] += float(emis)
    
# Function to process the emissions data efficiently using pandas
def process_emissions_data(emis_df, is_point):
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

    # If is_point is True, apply the safe_float_conversion function
    if is_point:
        emis_df['height'] = emis_df['stkhgt'].apply(safe_float_conversion, args=(0.3048, 0, 0, 5))
        emis_df['diam'] = emis_df['stkdiam'].apply(safe_float_conversion, args=(0.3048, 0, 0, 5))
        emis_df['temp'] = emis_df['stktemp'].apply(safe_float_conversion, args=(5.0/9.0, 0, 273.15 - (32 * 5.0/9.0), 5))
        emis_df['velocity'] = emis_df['stkvel'].apply(safe_float_conversion, args=(0.3048, 0, 0, 5))
        emis_df['coords'] = emis_df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)

    # Group by 'FIPS' and 'SCC' and aggregate the values
    aggregation_functions = {
        'region_cd': 'first',
        'scc': 'first',
        'VOC': 'sum',
        'NOx': 'sum',
        'NH3': 'sum',
        'SOx': 'sum',
        'PM2_5': 'sum'
    }

    if is_point:
        aggregation_functions.update({
            'height': 'first',
            'diam': 'first',
            'temp': 'first',
            'velocity': 'first',
            'facility_id': 'first',
            'coords': 'first'
        })

        # for point source, facility_id and scc is unique to each data point  
        grouped_emis_df = emis_df.groupby(['facility_id', 'scc']).agg(aggregation_functions).reset_index(drop=True)
        grouped_emis_df.rename(columns={'region_cd': 'FIPS','facility_id': 'EIS_ID', 'scc': 'SCC'}, inplace = True)
    else:
        # for non point source, fips and scc is unique to each data point  
        grouped_emis_df = emis_df.groupby(['region_cd', 'scc']).agg(aggregation_functions).reset_index(drop=True)
        grouped_emis_df.rename(columns={'region_cd': 'FIPS', 'scc': 'SCC'}, inplace = True)

    print("grouped_emis_df dataframe")
    print(grouped_emis_df.head())

    return grouped_emis_df

# Function to update the CSV file with new data
def save_state_emis(file_name, total_emissions, output_state_emis_csv):
    # Convert the dictionary to a DataFrame
    total_emissions_df = pd.DataFrame(total_emissions).T
    total_emissions_df.reset_index(inplace=True)
    total_emissions_df.columns = ['State', 'VOC', 'NOx', 'NH3', 'SOx', 'PM2_5']
    total_emissions_df['File'] = file_name

    # If the output CSV already exists, append to it
    if os.path.exists(output_state_emis_csv):
        existing_df = pd.read_csv(output_state_emis_csv)
        updated_df = pd.concat([existing_df, total_emissions_df], ignore_index=True)
    else:
        updated_df = total_emissions_df

    # Save the updated DataFrame to the CSV file
    updated_df.to_csv(output_state_emis_csv, index=False)


# this is a debuggin purpose only
def update_filtered_emis_df(emis_df, facility_id=6083511, filename='/Users/yunhalee/Documents/LOCAETA/CS_emissions/filtered_emis_df.csv'):

    # Filter the dataframe to find rows where 'facility_id' is 6083511
    filtered_df = emis_df[emis_df['facility_id'] == facility_id]
    
    print("Filtered EMIS DataFrame:")
    print(filtered_df.head())
    # Save the filtered dataframe to a CSV file to keep it updated
    filtered_df.to_csv(filename, index=False)
    
    return filtered_df

# this is a debuggin purpose only
def update_filtered_nei_df(nei_df, facility_id=6083511, filename='/Users/yunhalee/Documents/LOCAETA/CS_emissions/filtered_nei_df.csv'):

    # Filter the dataframe to find rows where 'facility_id' is 6083511
    filtered_df = nei_df[nei_df['EIS_ID'] == facility_id]
    
    print("Filtered NEI DataFrame:")
    print(filtered_df.head())
    # Save the filtered dataframe to a CSV file to keep it updated
    filtered_df.to_csv(filename, index=False)
    
    return filtered_df

# Function to check if a value is an integer
def is_integer(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


# Function to read and process a single NEI file
def process_nei_file(file_path):

    emis_df = pd.read_csv(file_path, comment='#', dtype={'region_cd': str})

    print(file_path, emis_df.columns)

    # Determine the type of emissions (point vs nonpoint) based on latitude/longitude columns existence
    is_point = set(['latitude', 'longitude']).issubset(emis_df.columns)

    if is_point :
        # Subset the dataframe to the columns we need
        emis_df = emis_df[['region_cd', 'scc', 'poll', 'ann_value', 
                        'stkhgt', 'stkdiam', 'stktemp', 'stkvel',
                        'facility_id', 
                        'latitude', 'longitude']]
    else:
        # Subset the dataframe to the columns we need
        emis_df = emis_df[['region_cd', 'scc', 'poll', 'ann_value']]

    print(emis_df.head())

    # debugging purpose
    #updated_filtered_emis_df = update_filtered_emis_df(emis_df)

    processed_df = process_emissions_data(emis_df, is_point)
    # debugging purpose
    #updated_filtered_nei_df = update_filtered_nei_df(processed_df)

    return processed_df, is_point


## The below functions are for processing CCS emissions

def reproject_and_save_gdf(gdf, target_crs):
    # Check if the GeoDataFrame is already in the target CRS
    current_crs = gdf.crs
    target_proj = CRS.from_string(target_crs)
    
    if current_crs != target_proj:
        print(f"Reprojecting from {current_crs} to {target_proj}")
        gdf = gdf.to_crs(target_crs)
    else:
        print(f"GeoDataFrame is already in the target CRS: {target_proj}")
    return gdf

def load_and_process_ccs_emissions_old(file_path):
    # This function processes Amy's old CS emission data file.
    cs_emis = pd.read_csv(file_path)
    all_columns = cs_emis.columns.tolist()
    start_index = all_columns.index("reporting_year")
    end_index = all_columns.index("FRS_ID")
    columns_to_keep = all_columns[:start_index + 1] + all_columns[end_index:]
    cs_emis = cs_emis[columns_to_keep]
    cs_emis.dropna(how='all', axis='columns', inplace=True)
    cs_emis.replace(["missing", "Blank"], np.nan, inplace=True)
    return cs_emis

NAN_FILL_VALUE = 0 # -9999

def load_and_process_ccs_emissions(file_path):
    # This function processes Kelly's new CS emission data file.
    cs_emis = pd.read_csv(file_path)

    # Filter columns based on partial name matching
    all_columns = cs_emis.columns
    columns_to_keep = ['scc']

    # Keep columns that end with '_id' or contain 'subpart_ton'
    for col in all_columns:
        if col.endswith('_id') or 'subpart_tons' in col:
            columns_to_keep.append(col)
    
    # Keep only the matching columns
    cs_emis = cs_emis[columns_to_keep]

    # Exclude the rows with missing SCC
    cs_emis = cs_emis[cs_emis['scc'] > NAN_FILL_VALUE] 
    
    # fill NA with zero
    cs_emis = cs_emis.fillna(NAN_FILL_VALUE)

    # Compute missing output from Kelly's file
    cs_emis['PM25_reduction_subpart_tons'] = cs_emis['PM25CON_reduction_subpart_tons']+ cs_emis['PM25FIL_reduction_subpart_tons']
    cs_emis['VOC_out_subpart_tons'] = cs_emis['VOC_subpart_tons']+ cs_emis['VOC_increase_subpart_tons']
    cs_emis['NH3_out_subpart_tons'] = cs_emis['NH3_subpart_tons']+ cs_emis['NH3_increase_subpart_tons']



    # ensure scc column to be integer
    cs_emis['scc'] = cs_emis['scc'].astype(int)

    cs_emis.rename(columns={'eis_id': 'EIS_ID', 'scc': 'SCC'}, inplace = True)

    return cs_emis

def fill_missing_eis_id(cs_emis, url):
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for file_name in z.namelist():
                if file_name.endswith('.csv'):
                    with z.open(file_name) as csv_file:
                        ID_df = pd.read_csv(csv_file)
                        ID_df = ID_df[["fips code", "eis facility id", "tri facility id", "company name", "site name", "site latitude", "site longitude"]]
                        ID_df.dropna(subset=["tri facility id"], inplace=True)
                        tri_to_eis = dict(zip(ID_df['tri facility id'], ID_df['eis facility id']))
                        cs_emis['EIS_ID'] = cs_emis.apply(
                            lambda row: row['EIS_ID'] if pd.notna(row['EIS_ID']) else tri_to_eis.get(row['TRI_ID'], row['EIS_ID']),
                            axis=1
                        )

                        # Check for any remaining missing EIS_ID
                        missing_eis_id_rows = cs_emis[cs_emis['EIS_ID'].isna()]

                        if not missing_eis_id_rows.empty:
                            raise ValueError(f"Missing EIS_ID for the following rows:\n{missing_eis_id_rows}")

                        print("All EIS_ID values are filled.")

                        return cs_emis
    else:
        raise RuntimeError(f"Failed to download Facility ZIP file: {response.status_code}")

    
def calculate_emission_rates_old(cs_emis):
    if cs_emis['EIS_ID'].duplicated().any():
        raise ValueError (f"CCS emissions EIS_ID has duplicates and needs attention")
    else: 
        cs_emis["PM_rate"] = np.where(cs_emis['facPrimaryPM25'] == 0, 0, (cs_emis['facPrimaryPM25'] - cs_emis['PM_red']) / cs_emis['facPrimaryPM25'])
        cs_emis["NOx_rate"] = np.where(cs_emis['facNOx'] == 0, 0, (cs_emis['facNOx'] - cs_emis['NOx_red']) / cs_emis['facNOx'])
        cs_emis["SO2_rate"] = np.where(cs_emis['facSO2'] == 0, 0, (cs_emis['facSO2'] - cs_emis['SO2_red']) / cs_emis['facSO2'])
        cs_emis["NH3_rate"] = np.where(cs_emis['facNH3'] == 0, 0, (cs_emis['facNH3'] + cs_emis['NH3_inc']) / cs_emis['facNH3'])
        cs_emis["VOC_rate"] = np.where(cs_emis['facVOC'] == 0, 0, (cs_emis['facVOC'] + cs_emis['VOCs_inc']) / cs_emis['facVOC'])

        # subset the cs_emis to the following columns
        cs_emis = cs_emis[ ["EIS_ID", "facPrimaryPM25", "facNOx" , "facSO2", "facNH3", "facVOC", "PM_red", "NOx_red" , "SO2_red", "NH3_inc", "VOCs_inc", "PM_rate", "NOx_rate" , "SO2_rate", "NH3_rate", "VOC_rate" ]]
        return cs_emis
    
def calculate_emission_rates(cs_emis):
    # This approach is not used anymore. 
    # Define pollutants with their calculation parameters
    pollutants = [
        ("PM25", "PM25_subpart_tons", "PM25_reduction_subpart_tons", -1),
        ("NOx", "NOX_subpart_tons", "NOX_reduction_subpart_tons", -1),
        ("SO2", "SO2_subpart_tons", "SO2_reduction_subpart_tons", -1),
        ("NH3", "NH3_subpart_tons", "NH3_increase_subpart_tons", 1),
        ("VOC", "VOC_subpart_tons", "VOC_increase_subpart_tons", 1)
    ]
    
    # Calculate all rates in one loop
    for poll, fac, adj, sign in pollutants:
        cs_emis[f"{poll}_rate"] = np.where(
            cs_emis[fac] == 0, 0, 
            (cs_emis[fac] + sign * cs_emis[adj]) / cs_emis[fac]
        )
    
    # Keep only required columns
    cols_to_keep = ["eis_id", "scc"]
    cols_to_keep.extend([p[1] for p in pollutants])  # Add fac columns
    cols_to_keep.extend([p[2] for p in pollutants])  # Add adj columns
    cols_to_keep.extend([f"{p[0]}_rate" for p in pollutants])  # Add rate columns
    
    return cs_emis[cols_to_keep]

def subset_and_validate_emissions(gdf, cs_emis):
    # First merge for the EIS_ID and SCC matched rows
    merged_df = pd.merge(gdf, cs_emis, on=['EIS_ID', 'SCC'], how='right')
    
    print(merged_df.columns)
    
    NEI_cols = ['VOC', 'NOx', 'NH3', 'SOx', 'PM2_5']
    CCS_cols = ['VOC_subpart_tons', 'NOX_subpart_tons', 'NH3_subpart_tons', 'SO2_subpart_tons', 'PM25_subpart_tons']
    
    # Check if NEI_cols and CCS_cols value matches
    for NEI, CCS in zip(NEI_cols, CCS_cols):
        # Check if columns exist in the dataframe
        if NEI in merged_df.columns and CCS in merged_df.columns:
            match_count = 0
            total_rows = len(merged_df)
            mismatch_examples = []
            
            # Iterate through each row to compare emissions
            for index, row in merged_df.iterrows():
                if row[CCS] == 0 and row[NEI] == 0:
                    match_count += 1
                elif row[CCS] == 0 and row[NEI] != 0:
                    mismatch_examples.append({
                        'index': index,
                        'EIS_ID': row.get('EIS_ID', 'N/A'),
                        'SCC': row.get('SCC', 'N/A'),
                        NEI: row[NEI],
                        CCS: row[CCS]
                    })
                else:
                    # Compare values
                    if (row[NEI] - row[CCS])/row[CCS] < 0.001 :
                        match_count += 1
                    else:
                        mismatch_examples.append({
                            'index': index,
                            'EIS_ID': row.get('EIS_ID', 'N/A'),
                            'SCC': row.get('SCC', 'N/A'),
                            NEI: row[NEI],
                            CCS: row[CCS]
                        })
            
            # Report matching results
            print(f"Comparing {NEI} with {CCS}:")
            print(f"  - Matches: {match_count} out of {total_rows} rows")
            
            # Show mismatches 
            if mismatch_examples:
                print("  - Mismatch examples:")
                for example in mismatch_examples:
                    print(f"    Row {example['index']} (EIS_ID: {example['EIS_ID']}, SCC: {example['SCC']}): {example[NEI]} vs {example[CCS]}")
        else:
            missing_cols = []
            if NEI not in merged_df.columns:
                missing_cols.append(NEI)
            if CCS not in merged_df.columns:
                missing_cols.append(CCS)
            print(f"Cannot compare {NEI} and {CCS}. Missing columns: {', '.join(missing_cols)}")
    
    merged_df.to_excel('/Users/yunhalee/Documents/LOCAETA/CS_emissions/validate_Kelly_NEI_emissions.xlsx', index=True)

    return merged_df

def merge_and_calculate_new_emissions_old(gdf, cs_emis):
    merged_df = pd.merge(gdf, cs_emis, on='EIS_ID', how='left')

    print(merged_df.columns)

    emissions_cols = ['VOC', 'NOx', 'NH3', 'SOx', 'PM2_5']
    rate_cols = ['VOC_rate', 'NOx_rate', 'NH3_rate', 'SO2_rate', 'PM25_rate']
    inc_cols = {'NH3': 'NH3_inc', 'VOC': 'VOCs_inc'}

    for emis, rate in zip(emissions_cols, rate_cols):
        if emis in inc_cols:
            inc = inc_cols[emis]

            def compute_new_emissions(row):
                if pd.notnull(row[rate]) and row[emis] != 0:
                    return row[emis] * row[rate]
                elif pd.notnull(row[inc]) and row[emis] == 0:
                    matching_rows = merged_df[
                        (merged_df['FIPS'] == row['FIPS']) & 
                        (merged_df[inc] == row[inc]) & 
                        (merged_df[emis] == 0)
                    ]
                    num_matching_rows = len(matching_rows)
                    return row[inc] / num_matching_rows if num_matching_rows > 0 else row[emis]
                else:
                    return row[emis]

            merged_df[emis + '_new'] = merged_df.apply(compute_new_emissions, axis=1)
        else:
            merged_df[emis + '_new'] = merged_df.apply(
                lambda row: row[emis] * row[rate] if pd.notnull(row[rate]) and row[emis] != 0 else row[emis], 
                axis=1
            )

    return merged_df

def merge_to_NEI_emissions(gdf, cs_emis):
    # First merge for the EIS_ID and SCC matched rows
    merged_df = pd.merge(gdf, cs_emis, on=['EIS_ID', 'SCC'], how='left')
    
    NEI_cols = ['VOC', 'NOx', 'NH3', 'SOx', 'PM2_5']
    CCS_cols = ['VOC_out_subpart_tons', 'NOX_out_subpart_tons', 'NH3_out_subpart_tons', 
                'SO2_out_subpart_tons', 'PM25_out_subpart_tons']
    
    for NEI, CCS in zip(NEI_cols, CCS_cols):
        # Create a mask for where CCS values are greater than NAN_FILL_VALUE
        valid_ccs_mask = merged_df[CCS] > NAN_FILL_VALUE
        
        # Create a new column with NEI values
        merged_df[NEI + '_new'] = merged_df[NEI]
        
        # Where the mask is True, replace with CCS values
        merged_df.loc[valid_ccs_mask, NEI + '_new'] = merged_df.loc[valid_ccs_mask, CCS]
    
    # Identify columns matching the pattern
    merged_df.drop(list(merged_df.filter(regex='subpart_tons')), axis=1, inplace=True)

    return merged_df


def plot_CCS_facility_emissions(df, output_dir):
    pollutants = ['VOC', 'NOx', 'NH3', 'SOx', 'PM2_5']
    pollutants_old = [f'{pollutant}_old' for pollutant in pollutants]

    totals_new = {pollutant: df[pollutant].sum() for pollutant in pollutants}
    totals_old = {f'{pollutant}': df[f'{pollutant}'].sum() for pollutant in pollutants_old}

    fig, axes = plt.subplots(nrows=len(pollutants), ncols=1, figsize=(20, 20))
    bar_width = 0.35
    for i, pollutant in enumerate(pollutants):
        ax = axes[i]
        indices = np.arange(len(df))
        ax.bar(indices, df[pollutant], bar_width, label=f'{pollutant}')
        ax.bar(indices + bar_width, df[pollutants_old[i]], bar_width, label=f'{pollutants_old[i]}')
        total_original = totals_old[f'{pollutant}_old']
        total_new = totals_new[f'{pollutant}']
        ax.set_title(f'{pollutant}\nTotal: {total_original:.3f} | New Total: {total_new:.3f} | Change: {(total_new - total_original):.3f} [tons]', fontsize=20)
        ax.set_xlabel('FIPS')
        ax.set_ylabel('Total emissions')
        ax.set_xticks(indices + bar_width / 2)
        ax.set_xticklabels(df['FIPS'], rotation=90)
        ax.legend()

    plt.tight_layout()
    plot_path = f"{output_dir}emissions_comparison.png"
    plt.savefig(plot_path)
    plt.close(fig)

    net_changes = [totals_new[f'{pollutant}'] - totals_old[f'{pollutant}_old'] for pollutant in pollutants]
    colors = ['blue' if val < 0 else 'red' for val in net_changes]
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(pollutants, net_changes, color=colors)
    ax.set_xlabel('Pollutants', fontsize=20)
    ax.set_ylabel('Net Change in Emissions (tons)', fontsize=20)
    ax.set_title('Net Changes in CCS Emissions by Pollutant', fontsize=30)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plot_path = f"{output_dir}net_changes.png"
    plt.savefig(plot_path)
    plt.close(fig)

import os
import sys
import warnings
import geopandas as gpd
import numpy as np

# Suppress all warnings
warnings.filterwarnings('ignore')

# Add the path to the main package directory
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'LOCAETA_AQ'))
if package_path not in sys.path:
    sys.path.append(package_path)

import inmap_analysis


# Load the census data
census_data = gpd.read_file("/Users/yunhalee/Documents/LOCAETA/RCM/INMAP/evaldata_v1.6.1/census2013blckgrp.shp")
print("census data", census_data.head())

# Load the mortality rate data
mortality_data = gpd.read_file("/Users/yunhalee/Documents/LOCAETA/RCM/INMAP/evaldata_v1.6.1/mortalityRates2013.shp")

print("mortality data", mortality_data.head())
# Extract relevant columns from census data
census_pop_columns = ["TotalPop", "WhiteNoLat", "Black", "Native", "Asian", "Latino"]
census_population = census_data[census_pop_columns]

# Map the mortality rate columns
mortality_rate_columns = {
    "TotalPop": "allcause", 
    "WhiteNoLat": "allcause", 
    "Black": "allcause", 
    "Native": "allcause", 
    "Asian": "allcause", 
    "Latino": "allcause"
}

INMAPOutputFile = '/Users/yunhalee/Documents/LOCAETA/RCM/INMAP/inmap-1.9.6/outputs/nei2020/2020nei_output_run_steady.shp'

# Load the INMAP output data
inmap_data = gpd.read_file(INMAPOutputFile)
print("INMAP data:", inmap_data.head())

# Perform a spatial join to add TotalPM25 data to the census data
census_data = census_data.to_crs(inmap_data.crs)  # Ensure both datasets have the same CRS
census_data = gpd.sjoin(census_data, inmap_data[['geometry', 'TotalPM25']], how='left', op='intersects')
print("Census data after spatial join:", census_data.head())

# Ensure both datasets have the same CRS
mortality_data = mortality_data.to_crs(census_data.crs)

# Perform a spatial join to add mortality rate data to the census data
census_data = gpd.sjoin(census_data, mortality_data[['geometry', 'allcause']], how='left', op='intersects')

# Display the first few rows to verify the join
print("Census data after adding mortality data:", census_data.head())


census_data['TotalPopD'] = (np.exp(np.log(1.078) / 10 * census_data['TotalPM25']) - 1) * census_data['TotalPop'] * mortality_rate / 100000
census_data['WhiteNoLatD'] = (np.exp(np.log(1.078) / 10 * census_data['TotalPM25']) - 1) * census_data['WhiteNoLat'] * mortality_rate / 100000
census_data['BlackD'] = (np.exp(np.log(1.078) / 10 * census_data['TotalPM25']) - 1) * census_data['Black'] * mortality_rate / 100000
census_data['NativeD'] = (np.exp(np.log(1.078) / 10 * census_data['TotalPM25']) - 1) * census_data['Native'] * mortality_rate / 100000
census_data['AsianD'] = (np.exp(np.log(1.078) / 10 * census_data['TotalPM25']) - 1) * census_data['Asian'] * mortality_rate / 100000
census_data['LatinoD'] = (np.exp(np.log(1.078) / 10 * census_data['TotalPM25']) - 1) * census_data['Latino'] * mortality_rate / 100000


print(census_data[['TotalPopD', 'WhiteNoLatD', 'BlackD', 'NativeD', 'AsianD', 'LatinoD']].head())

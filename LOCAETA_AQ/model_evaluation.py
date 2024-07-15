# model_evaluation.py

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import xarray as xr
from scipy.stats import linregress


def process_inmap(inmap_runs, inmap_output_path, inmap_variable, aqs_file_path, eval_output_dir, threshold, threshold_type):
    
    # Process and plot for INMAP each year
    for year, file in inmap_runs.items():
        inmap_run_output = os.path.join(inmap_output_path, file)
        aqs_file = os.path.join(aqs_file_path, f'annual_conc_by_monitor_{year}.csv')

        if not os.path.exists(inmap_run_output):
            print(f"File not found: {inmap_run_output}")
            continue

        if not os.path.exists(aqs_file):
            print(f"File not found: {aqs_file}")
            continue

        print(f"Processing files for year {year}:")
        print(f"inmap_run_output: {inmap_run_output}")
        print(f"aqs_file: {aqs_file}")

        # Create joined geodataframe with inmap and AQS data and unmodified geodataframe for inmap output
        joined_gdf, gdf = inmap_output_process(inmap_run_output, inmap_variable, aqs_file, year, eval_output_dir)

        # Create scatter plots between observation and model predictions
        plot_scatter_by_region(joined_gdf, inmap_variable, eval_output_dir, year, threshold=threshold, threshold_type=threshold_type)

        # Create scatter plots of model_to_obs ratio along with observation data (to detect any systemic bias)
        plot_scatter_ratio(joined_gdf, inmap_variable, eval_output_dir, year, threshold=threshold, threshold_type=threshold_type)

        # Create INMAP output map with AQS data 
        plot_map_from_geodf(joined_gdf, gdf, inmap_variable, eval_output_dir, year, threshold=threshold, threshold_type=threshold_type)

def process_satellite(satellites, satellite_netcdf_path,  satellite_variable, aqs_file_path, eval_output_dir, threshold, threshold_type):

    
    # Process and plot for Satellite PM25 each year
    for year, file in satellites.items():
        satellite_output = os.path.join(satellite_netcdf_path, file)
        aqs_file = os.path.join(aqs_file_path, f'annual_conc_by_monitor_{year}.csv')

        if not os.path.exists(satellite_output):
            print(f"File not found: {satellite_output}")
            continue

        if not os.path.exists(aqs_file):
            print(f"File not found: {aqs_file}")
            continue

        print(f"Processing files for year {year}:")
        print(f"satellite_output: {satellite_output}")
        print(f"aqs_file: {aqs_file}")

        # Create joined geodataframe with satellite and AQS data and unmodified geodataframe for satellite output
        joined_gdf, gdf = satellite_output_process(satellite_output, satellite_variable, aqs_file, year, eval_output_dir)

        # Create scatter plots between observation and model predictions
        plot_scatter_by_region(joined_gdf, satellite_variable, eval_output_dir, year, threshold=threshold, threshold_type=threshold_type)

        # Create scatter plots of model_to_obs ratio along with observation data (to detect any systemic bias)
        plot_scatter_ratio(joined_gdf, satellite_variable, eval_output_dir, year, threshold=threshold, threshold_type=threshold_type)

        # Create Satellite output map with AQS data 
        plot_map_from_netcdf(satellite_output, satellite_variable, aqs_file, eval_output_dir, year)
    
def inmap_output_process(shapefile_path, field, aqs_file_path, year, output_dir):

    gdf = gpd.read_file(shapefile_path)
    gdf = gdf[['geometry', field]]

    # Check and set the correct CRS for polygon data if not set correctly
    if gdf.crs is None:
        raise ValueError("INMAP output file doesn't have crs!")
        
    # Load EPA AQS annual PM2.5 observations
    data = pd.read_csv(aqs_file_path)

    # Filter the data for rows where 'Parameter Name' contains 'PM2.5 - Local Conditions'
    epa_aqs_data = data[data['Parameter Name'].str.contains('PM2.5 - Local Conditions', na=False)]

    # Select relevant columns and rename them for clarity
    epa_aqs_data = epa_aqs_data[['Latitude', 'Longitude', 'Arithmetic Mean']]
    epa_aqs_data.columns = ['latitude', 'longitude', 'pm25']

    # Remove duplicate rows based on 'latitude' and 'longitude'
    epa_aqs_data = epa_aqs_data.drop_duplicates(subset=['latitude', 'longitude'])

    print("epa data from dataframe :", year, len(epa_aqs_data))
    print( epa_aqs_data[:10])

    # Create GeoDataFrame for observation data
    geometry = [Point(xy) for xy in zip(epa_aqs_data['longitude'], epa_aqs_data['latitude'])]
    epa_gdf = gpd.GeoDataFrame(epa_aqs_data, geometry=geometry, crs='EPSG:4326')

    print("epa data from geo dataframe ")
    print(epa_gdf[:10])

    # Transform the AQS point data to match the CRS of the INMAP polygon data
    epa_gdf = epa_gdf.to_crs(gdf.crs)

    # Save the reprojected observation data to a new shapefile
    #reprojected_path = os.path.join(output_dir, f"annual_conc_by_monitor_{year}_reprojected.shp")
    #epa_gdf.to_file(reprojected_path)

    # Perform spatial join to sample model_pm25 from polygon shapefile and save them in point shapefile
    joined_gdf = epa_gdf.sjoin(gdf, how="inner", predicate='intersects')

    # Save the joined GeoDataFrame to a new shapefile
    output_joined_path = os.path.join(output_dir, f"annual_conc_by_monitor_with_model_{field}_{year}.shp")
    joined_gdf.to_file(output_joined_path)

    # Print debug information
    print("EPA AQS Data (Original):")
    print(epa_aqs_data[['latitude', 'longitude']].head())

    print("EPA AQS Data (Transformed):")
    print(epa_gdf[['geometry']].head())

    print("Joined Data:")
    print(joined_gdf.head())

    return joined_gdf, gdf

def naive_fast(latvar, lonvar, lat0, lon0):
    # this is when lat, lon are 2D
    latvals = latvar[:]
    lonvals = lonvar[:]
    dist_sq = (latvals - lat0)**2 + (lonvals - lon0)**2
    minindex_flattened = dist_sq.argmin()  # 1D index of min element
    iy_min, ix_min = np.unravel_index(minindex_flattened, latvals.shape)
    return int(iy_min), int(ix_min)

def find_closest(lat_array, lon_array, lat_point, lon_point):
    lat_diff = np.abs(lat_array - lat_point)
    lon_diff = np.abs(lon_array - lon_point)
    min_lat_idx = lat_diff.argmin()
    min_lon_idx = lon_diff.argmin()
    return min_lat_idx, min_lon_idx

def satellite_output_process(netcdf_file, field, aqs_file_path, year,output_dir):
    # Load the NetCDF file
    ds = xr.open_dataset(netcdf_file)

    # Extract latitude, longitude, and the variable data
    lat = ds['lat'].values
    lon = ds['lon'].values
    pm25_data = ds[field].values

    # Create a meshgrid of lat and lon
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lon_flat = lon_grid.flatten()
    lat_flat = lat_grid.flatten()
    pm25_flat = pm25_data.flatten()

    # Create a DataFrame
    df = pd.DataFrame({
        'latitude': lat_flat,
        'longitude': lon_flat,
        'GWRPM25': pm25_flat
    })

    # Load EPA AQS annual PM2.5 observations
    data = pd.read_csv(aqs_file_path)

    # Filter the data for rows where 'Parameter Name' contains 'PM2.5 - Local Conditions'
    epa_aqs_data = data[data['Parameter Name'].str.contains('PM2.5 - Local Conditions', na=False)]

    # Select relevant columns and rename them for clarity
    epa_aqs_data = epa_aqs_data[['Latitude', 'Longitude', 'Arithmetic Mean']]
    epa_aqs_data.columns = ['latitude', 'longitude', 'pm25']
    
    # Remove duplicate rows based on 'latitude' and 'longitude'
    epa_aqs_data = epa_aqs_data.drop_duplicates(subset=['latitude', 'longitude'])

    # Find nearest neighbors using naive_fast method
    nearest_pm25 = []
    for idx, row in epa_aqs_data.iterrows():
        iy, ix = find_closest(lat, lon, row['latitude'], row['longitude'])
        nearest_pm25.append(pm25_data[iy, ix])

    # Add the nearest PM2.5 data to the DataFrame
    epa_aqs_data[field] = nearest_pm25

    # Save the joined GeoDataFrame to a new shapefile
    output_joined_path = os.path.join(output_dir, f"annual_conc_by_monitor_with_satellite_{field}_{year}.csv")
    epa_aqs_data.to_csv(output_joined_path)

    return epa_aqs_data, df

# Define the regions with approximate latitude and longitude ranges
regions = {
    'Northeast': {'lat_range': [37, 47], 'lon_range': [-82, -67]},
    'Southeast': {'lat_range': [25, 37], 'lon_range': [-90, -75]},
    'Midwest': {'lat_range': [37, 47], 'lon_range': [-104, -82]},
    'Southwest': {'lat_range': [25, 37], 'lon_range': [-120, -104]},
    'West': {'lat_range': [37, 49], 'lon_range': [-125, -104]},
    'CONUS': {'lat_range': [25, 49], 'lon_range': [-125, -67]}
}

def plot_scatter_by_region(joined_gdf, field, output_dir, year, threshold=0, threshold_type ='more'):

    """
    Plot scatter based on a threshold.

    Parameters:
    - joined_gdf: GeoDataFrame containing the INMAP and AQS data
    - field: The field/column to apply the threshold on
    - output_dir: The directory to save the output
    - year: The year of the data
    - region : US geo regions defined in "regions" 
    - threshold: The threshold value applied to joined_gdf 
    - threshold_type: Determines if the threshold is 'less' or 'more' (default is 'more')
    """

    if threshold_type == 'less':
        # Filter data for values less than the threshold
        filtered_data = joined_gdf[joined_gdf[field] < threshold]
    elif threshold_type == 'more':
        # Filter data for values more than the threshold
        filtered_data = joined_gdf[joined_gdf[field] > threshold]
    else:
        raise ValueError("Invalid threshold_type. Use 'less' or 'more'.")

    for region in regions:
        region_gdf = filter_by_region(filtered_data, region)
        print("checking region", region, region_gdf.head())
        if not region_gdf.empty:
            plot_scatter(region_gdf, field, output_dir, year, region)
            
def filter_by_region(filtered_data, region):
    lat_range = regions[region]['lat_range']
    lon_range = regions[region]['lon_range']
    return filtered_data[(filtered_data['latitude'] >= lat_range[0]) & (filtered_data['latitude'] <= lat_range[1]) & 
               (filtered_data['longitude'] >= lon_range[0]) & (filtered_data['longitude'] <= lon_range[1])]

def plot_scatter(region_gdf, field, output_dir, year, region):

    # Calculate statistical measures
    observed = region_gdf['pm25']
    modeled = region_gdf[field]

    correlation_coefficient = np.corrcoef(observed, modeled)[0, 1]
    mean_bias = np.mean(modeled - observed)
    rmse = np.sqrt(mean_squared_error(observed, modeled))
    r_squared = r2_score(observed, modeled)
    nmb = np.sum(modeled - observed) / np.sum(observed) * 100
    nme = np.sum(np.abs(modeled - observed)) / np.sum(observed) * 100

    # Calculate Mean Error (ME)
    mean_error = np.mean(np.abs(modeled - observed))

    # Calculate Mean Fractional Bias (MFB)
    mean_fractional_bias = 2 * np.mean((modeled - observed) / (modeled + observed)) * 100

    # Calculate Mean Fractional Error (MFE)
    mean_fractional_error = 2 * np.mean(np.abs(modeled - observed) / (modeled + observed)) * 100

    # Calculate Model Ratio (MR)
    model_ratio = np.mean(modeled / observed)

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(observed, modeled)

    # Calculate Squared Pearson Correlation Coefficient (R²)
    r_squared = r_value ** 2

    # Plot scatter plot of observed vs. modeled PM2.5
    plt.figure(figsize=(8, 8))
    plt.scatter(observed, modeled, alpha=0.5)
    plt.plot([0, observed.max()], [0, observed.max()], 'r--')
    plt.xlabel('Observed PM2.5 (µg/m³)')
    plt.ylabel('Modeled PM2.5 (µg/m³)')
    plt.title(f'Observed vs. Modeled PM2.5 ({year}) - {region}')
    plt.grid(True)

    # Add statistical measures to the plot
    plt.text(0.05, 0.96, f'Correlation: {correlation_coefficient:.2f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(0.05, 0.92, f'Mean Bias: {mean_bias:.2f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(0.05, 0.88, f'Mean Error: {mean_error:.2f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(0.05, 0.84, f'Model Ratio : {model_ratio:.2f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(0.05, 0.80, f'R²: {r_squared:.2f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(0.05, 0.76, f'Slope: {slope:.2f}%', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(0.05, 0.72, f'Mean Fractional Bias: {mean_fractional_bias:.2f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(0.05, 0.68, f'Mean Fractional Error: {mean_fractional_error:.2f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.savefig(os.path.join(output_dir, f'observed_vs_model_{field}_{year}_{region}.png'), dpi=300, bbox_inches='tight')


def plot_scatter_ratio(joined_gdf, field, output_dir, year, threshold=0, threshold_type='more'):

    """
    Plot scatter ratio based on a threshold.

    Parameters:
    - joined_gdf: GeoDataFrame containing the INMAP and AQS data
    - field: The field/column to apply the threshold on
    - output_dir: The directory to save the output
    - year: The year of the data
    - threshold: The threshold value applied to joined_gdf 
    - threshold_type: Determines if the threshold is 'less' or 'more' (default is 'more')
    """
    if threshold_type == 'less':
        # Filter data for values less than the threshold
        filtered_data = joined_gdf[joined_gdf[field] < threshold]
    elif threshold_type == 'more':
        # Filter data for values more than the threshold
        filtered_data = joined_gdf[joined_gdf[field] > threshold]
    else:
        raise ValueError("Invalid threshold_type. Use 'less' or 'more'.")
    
    # Calculate statistical measures
    observed = filtered_data['pm25']
    modeled = filtered_data[field]

    # Sort the observed values and get the corresponding modeled values
    sorted_indices = np.argsort(observed)
    sorted_observed = observed.iloc[sorted_indices]
    sorted_modeled = modeled.iloc[sorted_indices]

    # Calculate the ratio of modeled to observed
    ratio = sorted_modeled / sorted_observed

    # Plot observed vs. ratio of modeled to observed
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_observed, ratio, 'o', alpha=0.5)
    plt.axhline(y=1, color='r', linestyle='--')  # Line at y=1 for reference
    plt.xlabel('Observed PM2.5 (µg/m³)')
    plt.ylabel('Modeled / Observed PM2.5 Ratio')
    plt.title(f'Observed vs. Modeled/Observed Ratio PM2.5 ({year})')
    plt.grid(True)

    # Save the plot
    plt.savefig(os.path.join(output_dir, f'observed_vs_model_{field}_ratio_{year}.png'), dpi=300, bbox_inches='tight')

def plot_map_from_geodf(joined_gdf, gdf, field, output_dir, year, threshold = 0, threshold_type='more'):
    # Define a color map for PM2.5
    color_map = mcolors.LinearSegmentedColormap.from_list(
        'custom_colormap',
        ['white', 'darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red']
    )

    # Normalize the color map to the range of values you want to display
    norm = mcolors.BoundaryNorm([0, 1, 3, 5, 7, 10, 20, 30, 50, 100], color_map.N)

    # Create a plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 10), subplot_kw={'projection': ccrs.LambertConformal()})

    # Plot polygons without the grid lines
    gdf.plot(column=field, cmap=color_map, linewidth=0.0, ax=ax, edgecolor='none', norm=norm, legend=False, transform=ccrs.LambertConformal(), alpha=0.9)


    if threshold_type == 'less':
        # Filter data for values less than the threshold
        filtered_data = joined_gdf[joined_gdf[field] < threshold]
    elif threshold_type == 'more':
        # Filter data for values more than the threshold
        filtered_data = joined_gdf[joined_gdf[field] > threshold]
    else:
        raise ValueError("Invalid threshold_type. Use 'less' or 'more'.")

    # Plot points as filled circles
    filtered_data.plot(ax=ax, markersize=25, marker='o', column='pm25', cmap=color_map, norm=norm, edgecolor='black', transform=ccrs.LambertConformal())

    # Add state and county borders for reference
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, linestyle=':', edgecolor='gray')

    # Color bar
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label('PM2.5 Concentration')

    # Titles and labels
    ax.set_title(f'PM2.5 Concentration from Model and EPA Observations ({year})', fontsize=16)

    # Set the extent to cover the CONUS region
    ax.set_extent([-125, -66.5, 24, 50], crs=ccrs.PlateCarree())

    # Save the plot to a file
    plt.savefig(os.path.join(output_dir, f'INMAP_{field}_map_with_AQS_{year}.png'), dpi=300, bbox_inches='tight')


def plot_map_from_netcdf(netcdf_file, field, aqs_file_path, output_dir, year):
    # Define a color map for PM2.5
    color_map = mcolors.LinearSegmentedColormap.from_list(
        'custom_colormap',
        ['white', 'darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red']
    )

    # Load the NetCDF file
    ds = xr.open_dataset(netcdf_file)

    # Extract latitude, longitude, and the variable data
    lat = ds['lat'].values
    lon = ds['lon'].values
    pm25_data = ds[field].values

    # Load EPA AQS annual PM2.5 observations
    data = pd.read_csv(aqs_file_path)

    # Filter the data for rows where 'Parameter Name' contains 'PM2.5 - Local Conditions'
    epa_aqs_data = data[data['Parameter Name'].str.contains('PM2.5 - Local Conditions', na=False)]

    # Select relevant columns and rename them for clarity
    epa_aqs_data = epa_aqs_data[['Latitude', 'Longitude', 'Arithmetic Mean']]
    epa_aqs_data.columns = ['latitude', 'longitude', 'pm25']

    # Remove duplicate rows based on 'latitude' and 'longitude'
    epa_aqs_data = epa_aqs_data.drop_duplicates(subset=['latitude', 'longitude'])

    # Create GeoDataFrame for observation data
    geometry = [Point(xy) for xy in zip(epa_aqs_data['longitude'], epa_aqs_data['latitude'])]
    epa_gdf = gpd.GeoDataFrame(epa_aqs_data, geometry=geometry, crs='EPSG:4326')

    # Set up the plot with a specific projection
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Add features to the plot
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES, linestyle=':')

    # Plot the PM2.5 data using pcolormesh
    mesh = ax.pcolormesh(lon, lat, pm25_data, transform=ccrs.PlateCarree(), cmap=color_map, shading='auto')

    # Plot EPA AQS observation data
    scatter = ax.scatter(
        epa_gdf['longitude'],
        epa_gdf['latitude'],
        c=epa_gdf['pm25'],
        cmap=color_map,
        edgecolor='black',
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
        s=50,  # size of the circles
        label='EPA AQS Observations'
    )

    # Add a color bar
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_label(f'{field} (ug m-3)')

    # Set the extent to cover the CONUS region
    ax.set_extent([-125, -66.5, 24, 50], crs=ccrs.PlateCarree())

    # Add title
    plt.title(f'{field} over the USA with EPA AQS Observations')

    # Save the plot to a file
    plt.savefig(os.path.join(output_dir, f'satellite_{field}_map_with_AQS_{year}.png'), dpi=300, bbox_inches='tight')

    # Show the plot
    #plt.show()
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np


def calculate_county_intersected_areas(inmap, gdf_fips):
    # Calculate the area of each grid cell
    inmap['grid_cell_area'] = inmap.geometry.area
    
    # Perform an intersection to calculate intersected areas
    inmap_with_fips = gpd.overlay(inmap, gdf_fips[['FIPS', 'geometry']], how='intersection')
    
    # Calculate the intersected area
    inmap_with_fips['intersected_area'] = inmap_with_fips.geometry.area
    
    return inmap_with_fips

def reshape_inmap_to_county_data(inmap_with_fips, gdf_fips):
    # Calculate the area-weighted TotalPM25
    inmap_with_fips['weighted_TotalPM25'] = inmap_with_fips['TotalPM25'] * inmap_with_fips['intersected_area']
    
    # Aggregate data at the county level
    county_level_data = inmap_with_fips.groupby('FIPS').agg({
        'weighted_TotalPM25': 'sum',
        'intersected_area': 'sum'
    }).reset_index()
    
    # Calculate the area-weighted average TotalPM25
    county_level_data['area_weighted_avg_TotalPM25'] = county_level_data['weighted_TotalPM25'] / county_level_data['intersected_area']
    
    # Calculate the actual area for each county
    gdf_fips['county_area'] = gdf_fips.geometry.area
    
    # Merge the aggregated data with the actual county areas
    county_level_data = county_level_data.merge(gdf_fips[['FIPS', 'county_area', 'geometry']], on='FIPS', how='left')
    
    # Compare intersected_area with actual county_area
    county_level_data['area_difference'] = county_level_data['intersected_area'] - county_level_data['county_area']
    
    return gpd.GeoDataFrame(county_level_data, geometry='geometry')

def save_AQ_csv(county_level_gdf, output_csv_path):
    # Prepare data for CSV
    county_level_gdf['Annual Metric'] = 'Mean'
    inmap_csv = county_level_gdf[['Row', 'Column', 'Annual Metric']]
    inmap_csv['Values'] = county_level_gdf['area_weighted_avg_TotalPM25']
    
    # Save as CSV
    inmap_csv.to_csv(output_csv_path, index=False)

def save_grid_shapefile(county_level_gdf, output_shapefile_path):
    # Save as new shapefile
    county_level_grid = county_level_gdf[['Row', 'Column', 'geometry']]
    county_level_grid.to_file(output_shapefile_path)

def process_inmap_to_benmap_inputs(inmap_output_path, county_shapefile_path, output_shapefile_path, output_csv_path):

    inmap = gpd.read_file(inmap_output_path)
    inmap = inmap[['geometry', 'TotalPM25']]
    
    gdf_fips = gpd.read_file(county_shapefile_path)
    gdf_fips['FIPS'] = gdf_fips['STATEFP'].astype(str) + gdf_fips['COUNTYFP'].astype(str)

    print("CRS of inmap:", inmap.crs)
    print("CRS of gdf_fips before conversion:", gdf_fips.crs)
    
    # BenMAP's preferred projection for area calculation
    projection = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=23, standard_parallels=(29.5, 45.5))
    
    gdf_fips =  gdf_fips.to_crs(projection) 
    inmap = inmap.to_crs(projection) 
    
    print("CRS of inmap after geographic conversion:", inmap.crs)
    print("CRS of gdf_fips after geographic conversion:", gdf_fips.crs)
    
    # Compute the intersection area for area-weighted concentrations
    inmap_with_fips = calculate_county_intersected_areas(inmap, gdf_fips)
    
    # Reshape INMAP grids to county-level 
    county_level_gdf = reshape_inmap_to_county_data(inmap_with_fips, gdf_fips)
    
    # Calculate centroids
    county_level_gdf['centroid'] = county_level_gdf.geometry.centroid
    county_level_gdf['centroid_x'] = county_level_gdf.centroid.x
    county_level_gdf['centroid_y'] = county_level_gdf.centroid.y
    
    # Sort by latitude (northing) then by longitude (easting)
    county_level_gdf = county_level_gdf.sort_values(by=['centroid_y', 'centroid_x'])
    
    # Assign Row and Column sequentially
    county_level_gdf = county_level_gdf.reset_index(drop=True)
    county_level_gdf['Row'] = (county_level_gdf.index // county_level_gdf.shape[1]) + 1
    county_level_gdf['Column'] = (county_level_gdf.index % county_level_gdf.shape[1]) + 1
    
    # Save shapefile and CSV
    save_grid_shapefile(county_level_gdf, output_shapefile_path)
    save_AQ_csv(county_level_gdf, output_csv_path)
    
    print(f"Shapefile and CSV for {inmap_output_path} have been saved successfully.")
    
    return inmap, county_level_gdf

def plot_pm25_original_and_reshaped_results(inmap, county_level_gdf, output_prefix, output_dir):

    # Debugging: Print the min and max of the PM2.5 values
    print("InMAP TotalPM25 min:", inmap['TotalPM25'].min(), "max:", inmap['TotalPM25'].max())
    print("County-level PM2.5 min:", county_level_gdf['area_weighted_avg_TotalPM25'].min(), "max:", county_level_gdf['area_weighted_avg_TotalPM25'].max())
    
    # Calculate statistics for InMAP
    inmap_min = inmap['TotalPM25'].min()
    inmap_max = inmap['TotalPM25'].max()
    inmap_median = inmap['TotalPM25'].median()
    inmap_std = inmap['TotalPM25'].std()
    inmap_mean = np.average(inmap['TotalPM25'], weights=inmap['grid_cell_area'])
    
    # Calculate statistics for County Level
    county_min = county_level_gdf['area_weighted_avg_TotalPM25'].min()
    county_max = county_level_gdf['area_weighted_avg_TotalPM25'].max()
    county_median = county_level_gdf['area_weighted_avg_TotalPM25'].median()
    county_std = county_level_gdf['area_weighted_avg_TotalPM25'].std()
    county_mean = np.average(county_level_gdf['area_weighted_avg_TotalPM25'], weights=county_level_gdf['intersected_area'])

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), subplot_kw={'projection': ccrs.LambertConformal()})

    inmap.plot(
        column='TotalPM25',
        cmap='Reds',
        legend=True,
        ax=axes[0],  # First row, first column
        vmin=0,
        vmax=30,
        transform=ccrs.LambertConformal()
    )
    axes[0].set_title('InMAP Output: TotalPM25')

    county_level_gdf = county_level_gdf.to_crs(ccrs.LambertConformal())
    county_level_gdf.plot(
        column='area_weighted_avg_TotalPM25',
        cmap='Reds',
        legend=True,
        ax=axes[1],  # First row, second column
        vmin=0,
        vmax=30,
        transform=ccrs.LambertConformal()
    )
    axes[1].set_title('County Level: Area Weighted Avg TotalPM25')

    plt.tight_layout()
    plt.savefig(f"{output_dir}{output_prefix}_PM25_maps.png")
    #plt.show()

    # Calculate the bin edges for histograms
    bin_edges = np.linspace(0, 30, 31)  # 30 bins from 0 to 50

    # Define figure and axes for histograms
    fig_hist, ax_hist = plt.subplots(1, 2, figsize=(15, 7))

    inmap_values = inmap['TotalPM25'].values
    ax_hist[0].hist(inmap_values, bins=bin_edges, color='skyblue', alpha=0.7, rwidth=0.8)
    ax_hist[0].set_title('InMAP PM2.5 Distribution')
    ax_hist[0].set_xlabel('Total PM2.5')
    ax_hist[0].set_ylabel('Frequency')
    ax_hist[0].set_xlim(0, 50)
    ax_hist[0].grid(axis='y', alpha=0.5)
    ax_hist[0].axvline(inmap_median, color='orange', linestyle='dashed', linewidth=1)
    ax_hist[0].axvline(inmap_mean, color='red', linestyle='dashed', linewidth=1)
    textstr = f'Median: {inmap_median:.2f}\nMean: {inmap_mean:.2f}\nStd: {inmap_std:.2f}'
    ax_hist[0].text(0.95, 0.95, textstr, transform=ax_hist[0].transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right', 
                    bbox=dict(facecolor='white', alpha=0.5))

    county_values = county_level_gdf['area_weighted_avg_TotalPM25'].values
    ax_hist[1].hist(county_values, bins=bin_edges, color='lightcoral', alpha=0.7, rwidth=0.8)
    ax_hist[1].set_title('County-level PM2.5 Distribution')
    ax_hist[1].set_xlabel('Area Weighted Avg Total PM2.5')
    ax_hist[1].set_ylabel('Frequency')
    ax_hist[1].set_xlim(0, 50)
    ax_hist[1].grid(axis='y', alpha=0.5)
    ax_hist[1].axvline(county_median, color='orange', linestyle='dashed', linewidth=1)
    ax_hist[1].axvline(county_mean, color='red', linestyle='dashed', linewidth=1)
    textstr = f'Median: {county_median:.2f}\nMean: {county_mean:.2f}\nStd: {county_std:.2f}'
    ax_hist[1].text(0.95, 0.95, textstr, transform=ax_hist[1].transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right', 
                    bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{output_dir}{output_prefix}_comparison.png")
    #plt.show()
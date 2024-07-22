import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np


def calculate_grid_intersected_areas(inmap, gdf_grids):
    # Calculate the area of each grid cell
    inmap['grid_cell_area'] = inmap.geometry.area
    
    # Perform an intersection to calculate intersected areas
    inmap_with_fips = gpd.overlay(inmap, gdf_grids[['ID', 'geometry']], how='intersection')

    # Calculate the intersected area
    inmap_with_fips['intersected_area'] = inmap_with_fips.geometry.area
    
    return inmap_with_fips

def reshape_inmap_to_grid_data(inmap_with_fips, gdf_grids):
    # Calculate the area-weighted TotalPM25
    inmap_with_fips['weighted_TotalPM25'] = inmap_with_fips['TotalPM25'] * inmap_with_fips['intersected_area']
    
    # Aggregate data at the county level
    grid_level_data = inmap_with_fips.groupby('ID').agg({
        'weighted_TotalPM25': 'sum',
        'intersected_area': 'sum'
    }).reset_index()
    
    # Calculate the area-weighted average TotalPM25
    grid_level_data['area_weighted_avg_TotalPM25'] = grid_level_data['weighted_TotalPM25'] / grid_level_data['intersected_area']
    
    # Calculate the actual area for each county
    gdf_grids['grid_area'] = gdf_grids.geometry.area
    
    # Merge the aggregated data with the actual county areas
    grid_level_data = grid_level_data.merge(gdf_grids[['ID', 'ROW', 'COL', 'grid_area', 'geometry']], on='ID', how='left')
    
    # Compare intersected_area with actual grid_area
    grid_level_data['area_difference'] = grid_level_data['intersected_area'] - grid_level_data['grid_area']
    
    return gpd.GeoDataFrame(grid_level_data, geometry='geometry')

def save_AQ_csv(grid_level_gdf, output_csv_path):
    # Prepare data for CSV
    grid_level_gdf['Annual Metric'] = 'Mean'
    inmap_csv = grid_level_gdf[['ROW', 'COL', 'Annual Metric','area_weighted_avg_TotalPM25']]
    # Rename the columns name to BenMAP format
    inmap_csv = inmap_csv.rename(columns={'ROW': 'Row', 'COL':'Column','area_weighted_avg_TotalPM25':'Values'})
    
    # Save as CSV
    inmap_csv.to_csv(output_csv_path, index=False)

def save_grid_shapefile(grid_level_gdf, output_shapefile_path):
    # Save as new shapefile
    grid_level_grid = grid_level_gdf[['ROW', 'COL', 'geometry','area_weighted_avg_TotalPM25']]
    grid_level_grid = grid_level_grid.rename(columns={'ROW': 'Row', 'COL':'Column'})
    grid_level_grid.to_file(output_shapefile_path)

def process_inmap_to_benmap_inputs(inmap_output_path, grid_shapefile_path, output_shapefile_path, output_csv_path, grid_level):

    inmap = gpd.read_file(inmap_output_path)
    inmap = inmap[['geometry', 'TotalPM25']]
    
    gdf_grids = gpd.read_file(grid_shapefile_path)

    if grid_level == 'tract':
        # Create a unique ID for tracts using ROW and COL
        gdf_grids['ID'] = gdf_grids['ROW'].astype(str) + '_' + gdf_grids['COL'].astype(str)
    elif grid_level == 'county':  
        gdf_grids = gdf_grids.rename(columns={'FIPS': 'ID'})
        

    print("CRS of inmap:", inmap.crs)
    print("CRS of gdf_grids before conversion:", gdf_grids.crs)
    
    # BenMAP's preferred projection for area calculation
    projection = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=23, standard_parallels=(29.5, 45.5))
    
    gdf_grids =  gdf_grids.to_crs(projection) 
    inmap = inmap.to_crs(projection) 
    
    print("CRS of inmap after geographic conversion:", inmap.crs)
    print("CRS of gdf_grids after geographic conversion:", gdf_grids.crs)
    
    # Compute the intersection area for area-weighted concentrations
    inmap_with_grids = calculate_grid_intersected_areas(inmap, gdf_grids)
    
    # Reshape INMAP grids to grid level
    grid_level_gdf = reshape_inmap_to_grid_data(inmap_with_grids, gdf_grids)
    

    
    '''
    # Calculate centroids
    grid_level_gdf['centroid'] = grid_level_gdf.geometry.centroid
    grid_level_gdf['centroid_x'] = grid_level_gdf.centroid.x
    grid_level_gdf['centroid_y'] = grid_level_gdf.centroid.y
    
    # Sort by latitude (northing) then by longitude (easting)
    grid_level_gdf = grid_level_gdf.sort_values(by=['centroid_y', 'centroid_x'])
    
    # Assign Row and Column sequentially
    grid_level_gdf = grid_level_gdf.reset_index(drop=True)
    grid_level_gdf['Row'] = (grid_level_gdf.index // grid_level_gdf.shape[1]) + 1
    grid_level_gdf['Column'] = (grid_level_gdf.index % grid_level_gdf.shape[1]) + 1
    '''

    # Save shapefile and CSV
    save_grid_shapefile(grid_level_gdf, output_shapefile_path)
    save_AQ_csv(grid_level_gdf, output_csv_path)
    
    print(f"Shapefile and CSV for {inmap_output_path} have been saved successfully.")
    
    return inmap, grid_level_gdf

def plot_pm25_original_and_reshaped_results(inmap, grid_level_gdf, output_prefix, output_dir, grid_level):

    # Debugging: Print the min and max of the PM2.5 values
    print("InMAP TotalPM25 min:", inmap['TotalPM25'].min(), "max:", inmap['TotalPM25'].max())
    print("Grid-level PM2.5 min:", grid_level_gdf['area_weighted_avg_TotalPM25'].min(), "max:", grid_level_gdf['area_weighted_avg_TotalPM25'].max())
    
    # Calculate statistics for InMAP
    inmap_min = inmap['TotalPM25'].min()
    inmap_max = inmap['TotalPM25'].max()
    inmap_median = inmap['TotalPM25'].median()
    inmap_std = inmap['TotalPM25'].std()
    inmap_mean = np.average(inmap['TotalPM25'], weights=inmap['grid_cell_area'])
    
    # Calculate statistics for Grid Level
    grid_min = grid_level_gdf['area_weighted_avg_TotalPM25'].min()
    grid_max = grid_level_gdf['area_weighted_avg_TotalPM25'].max()
    grid_median = grid_level_gdf['area_weighted_avg_TotalPM25'].median()
    grid_std = grid_level_gdf['area_weighted_avg_TotalPM25'].std()
    grid_mean = np.average(grid_level_gdf['area_weighted_avg_TotalPM25'], weights=grid_level_gdf['intersected_area'])

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

    grid_level_gdf = grid_level_gdf.to_crs(ccrs.LambertConformal())
    grid_level_gdf.plot(
        column='area_weighted_avg_TotalPM25',
        cmap='Reds',
        legend=True,
        ax=axes[1],  # First row, second column
        vmin=0,
        vmax=30,
        transform=ccrs.LambertConformal()
    )
    axes[1].set_title('Grid Level: Area Weighted Avg TotalPM25')

    plt.tight_layout()
    plt.savefig(f"{output_dir}{output_prefix}_{grid_level}_PM25_maps.png")
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

    grid_values = grid_level_gdf['area_weighted_avg_TotalPM25'].values
    ax_hist[1].hist(grid_values, bins=bin_edges, color='lightcoral', alpha=0.7, rwidth=0.8)
    ax_hist[1].set_title('Grid-level PM2.5 Distribution')
    ax_hist[1].set_xlabel('Area Weighted Avg Total PM2.5')
    ax_hist[1].set_ylabel('Frequency')
    ax_hist[1].set_xlim(0, 50)
    ax_hist[1].grid(axis='y', alpha=0.5)
    ax_hist[1].axvline(grid_median, color='orange', linestyle='dashed', linewidth=1)
    ax_hist[1].axvline(grid_mean, color='red', linestyle='dashed', linewidth=1)
    textstr = f'Median: {grid_median:.2f}\nMean: {grid_mean:.2f}\nStd: {grid_std:.2f}'
    ax_hist[1].text(0.95, 0.95, textstr, transform=ax_hist[1].transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right', 
                    bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{output_dir}{output_prefix}_{grid_level}_comparison.png")
    #plt.show()
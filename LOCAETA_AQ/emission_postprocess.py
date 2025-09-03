import contextily as ctx
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import pandas as pd

def plot_spatial_distribution_percent_change_with_basemap(gdf, output_dir, national_scale = False):
    pollutants = ['NH3', 'VOC', 'NOx', 'SOx', 'PM2_5']
    
    # Ensure the GeoDataFrame is in the correct CRS for basemaps
    gdf = gdf.to_crs(epsg=3857)
    
    # Collect stats here
    stats = []

    for pollutant in pollutants:
        # Use gridspec for more precise layout control
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(1, 20)  # Create a 1×20 grid for precise width control
        
        # Main plot takes 18/20 of the width
        ax = fig.add_subplot(gs[0, :19])
        
        col_current = f'{pollutant}'
        col_old = f'{pollutant}_nei'
        
        if col_current not in gdf.columns or col_old not in gdf.columns:
            print(f'Columns {col_current} or {col_old} do not exist in the data.')
            continue
        
        # Calculate percent change
        gdf['percent_change'] = ((gdf[col_current] - gdf[col_old]) / gdf[col_old].replace(0, float('nan'))) * 100
        
        # debugging
        gdf[gdf['percent_change'] > 0].to_csv(output_dir + 'debuggin_USA_positive_PM_changes.csv', index=False)
        
        # Set color scale
        vmin, vmax = -100, 100
        if pollutant == 'NH3':
            vmin, vmax = -200, 200
        
        # Plot data
        gdf.plot(column='percent_change', cmap='coolwarm', vmin=vmin, vmax=vmax, 
                legend=False, edgecolor='black', ax=ax, markersize=30, alpha=0.95)
        
        if national_scale:
            # Add basemap
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=4)
        else:   
            # Add basemap
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=10)
            
        # Set title
        ax.set_title(f'Percent Change in {pollutant} emissions by amine-based CCS')
        
        # Calculate statistics
        total_current = gdf[col_current].sum()
        total_old = gdf[col_old].sum()
        total_percent_change = ((total_current - total_old) / total_old) * 100
        max_percent_change = gdf['percent_change'].max()
        min_percent_change = gdf['percent_change'].min()
        

        # Save stats
        stats.append({
            'Pollutant': pollutant,
            'Total Percent Change [%]': round(total_percent_change,0)
            #'Max % Change': max_percent_change,
            #'Min % Change': min_percent_change
        })

        # Display statistics
        ax.text(0.5, -0.15, f'Total Percent Change: {total_percent_change:.2f}%\nMax Percent Change: {max_percent_change:.2f}%\nMin Percent Change: {min_percent_change:.2f}%',
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='black')
        
        # Create colorbar in the last 1/20 columns of the grid
        cbar_ax = fig.add_subplot(gs[0, 19:])
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax))
        sm._A = []
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(f'{pollutant} Percent Change', rotation=270, labelpad=15)
        
        # Minimize spacing between elements
        plt.subplots_adjust(wspace=0.05)  # Very small spacing between map and colorbar
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'{pollutant}_percent_change_with_basemap.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Convert stats to DataFrame
    stats_df = pd.DataFrame(stats)
    return stats_df

# Function to plot the percent change of each pollutant and its "_old" version with a basemap
def plot_spatial_distribution_relative_difference_with_basemap(gdf, output_dir, national_scale=False):
    pollutants = ['NH3', 'VOC', 'NOx',  'SOx', 'PM2_5'] #  
    
    # Ensure the GeoDataFrame is in the correct CRS for basemaps (Web Mercator)
    gdf = gdf.to_crs(epsg=3857)

    for pollutant in pollutants:
        # Use gridspec for more precise layout control
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(1, 20)  # Create a 1×20 grid for precise width control

        # Main plot takes 18/20 of the width
        ax = fig.add_subplot(gs[0, :19])
        
        col_current = f'{pollutant}'
        col_old = f'{pollutant}_nei'

        # Ensure both the current and "_old" columns exist in the GeoDataFrame
        if col_current not in gdf.columns or col_old not in gdf.columns:
            print(f'Columns {col_current} or {col_old} do not exist in the data.')
            continue
        
        # Calculate the percent change, avoiding division by zero
        gdf['reverse_percent_change'] = ((gdf[col_old] - gdf[col_current]) / gdf[col_current].replace(0, float('nan'))) * 100

        # Plot the spatial distribution of the percent change
        vmin, vmax = -100, 100  # Fixed color scale from -50% to 50%

        if pollutant == 'NH3':
            vmin, vmax = -200, 200 

        gdf.plot(column='reverse_percent_change', cmap='coolwarm', vmin=vmin, vmax=vmax, legend=False, edgecolor='black', 
                 ax=ax, markersize=30, alpha=0.95)  # Increase marker size and reduce transparency

        if national_scale:
            # Add basemap
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=4)
        else:   
            # Add basemap
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=10)

        ax.set_title(f'Relative Difference in {pollutant} NEI 2020 Emissions Compared to Amine-Based CCS Emissions')

        # Calculate the total, max, and min percent change
        total_current = gdf[col_current].sum()
        total_old = gdf[col_old].sum()
        total_percent_change = ((total_old - total_current) / total_current) * 100
        max_percent_change = gdf['reverse_percent_change'].max()
        min_percent_change = gdf['reverse_percent_change'].min()

        # Display the total, max, and min percent changes on the plot
        ax.text(0.5, -0.15, f'Total Relative Difference: {total_percent_change:.2f}%\nMax Relative Difference: {max_percent_change:.2f}%\nMin Relative Difference: {min_percent_change:.2f}%', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='black')

        # Create colorbar in the last 1/20 columns of the grid
        cbar_ax = fig.add_subplot(gs[0, 19:])
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax))
        sm._A = []
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(f'{pollutant} Relative Difference', rotation=270, labelpad=15)
        
        # Minimize spacing between elements
        plt.subplots_adjust(wspace=0.05)  # Very small spacing between map and colorbar

        # Save the figure for each pollutant as a separate file
        plt.savefig(os.path.join(output_dir, f'{pollutant}_Relative_Difference_with_basemap.png'), dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to avoid overlapping plots


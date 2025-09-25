import geopandas as gpd
import pandas as pd
import os
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
import matplotlib.colors as mcolors
import logging
from .run_benmap_utils import Benmap_Processor 
import json

# Suppress all warnings in jupyter notebook
import warnings
warnings.filterwarnings('ignore')

# logging from run_workflow 
logger = logging.getLogger(__name__)


class Benmap_Analyzer: 
    """
    A class to analyze BenMAP results 
    """

    def __init__(self, cfg):
        """
        Initialize the processor with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with file paths and parameters
        """
        self.cfg = cfg

    # Function to subset data for a specific state or use national data
    def subset_data(self, final_df, state_fips=None):
        if state_fips:
            # Ensure state_fips is a list
            if not isinstance(state_fips, list):
                state_fips = [state_fips]
            
            # Collect all states of interest, including neighbors
            all_fips = set(state_fips)

            # Filter the DataFrame
            return final_df[final_df['STATE_FIPS'].isin(all_fips)]
        return final_df


    def process_benmap_output(self, benmap_output_file, grid_gdf, benmap_output):
        """
        Process BenMAP output CSV and merge it with spatial grid data.

        Parameters
        ----------
        benmap_output_file : str
            Path to the BenMAP output CSV file.
        grid_gdf : geopandas.GeoDataFrame
            GeoDataFrame containing spatial grid information with 'Row' and 'Col' columns.
        benmap_output : str, optional
            Type of BenMAP output. Must be either 'incidence' or 'valuation'

        Returns
        -------
        geopandas.GeoDataFrame
            Merged and processed GeoDataFrame with cleaned columns and standardized endpoint labels.
        """
        # Read BenMAP output
        benmap = pd.read_csv(benmap_output_file)

        # Adjust for GUI-based output column naming
        benmap.rename(columns={"Column": "Col"}, inplace=True)
        benmap['Pollutant'] = 'PM2.5'

        # Merge with grid shapefile (left join on Row, Col)
        merged_df = benmap.merge(grid_gdf, on=['Row', 'Col'], how='left')

        # Select relevant columns
        if benmap_output == 'incidence':
            columns_to_keep = [
                'Endpoint', 'Pollutant', 'Author', 'Race', 'Ethnicity',
                'STATE_FIPS', 'CNTY_FIPS', 'Row', 'Col', 'Mean', 'Population', 'geometry'
            ]
        else:
            columns_to_keep = [
                'Endpoint', 'Pollutant', 'Author', 'Race', 'Ethnicity',
                'STATE_FIPS', 'CNTY_FIPS', 'Row', 'Col', 'Mean', 'geometry'
            ]

        final_df = merged_df[columns_to_keep].copy()

        # Reassign mortality endpoint names based on Author
        final_df.loc[final_df['Author'].str.contains('Pope', na=False), 'Endpoint'] = 'Mortality All Cause : Method 2'
        final_df.loc[final_df['Author'].str.contains('Di', na=False), 'Endpoint'] = 'Mortality All Cause : Method 1'

        # Reassign Hispanic from Ethnicity to Race
        final_df.loc[final_df['Ethnicity'].str.contains('HISPANIC', na=False), 'Race'] = 'HISPANIC'

        # Drop Ethnicity column
        final_df = final_df.drop(columns=['Ethnicity'])

        final_df = gpd.GeoDataFrame(final_df, geometry= "geometry")
        return final_df

    # Example function to format values
    def format_values(self, value):
        return f'{value:.4g}' if isinstance(value, (int, float)) else value

    # Function to create and save a table as CSV
    def create_csv(self, df, columns, title, output_dir):
        # Format the values in the DataFrame
        formatted_df = df[columns].applymap(self.format_values)
        
        # Create the file path
        file_path = os.path.join(output_dir, f'{title}.csv')
        
        # Save the DataFrame to CSV
        formatted_df.to_csv(file_path, index=False)
        
        logger.info(f"CSV file saved at {file_path}")

    def plot_spatial_distribution_benmap_with_basemap(self, gdf, field, output_dir, region_name):

        # Define the preferred race order
        race_order = ["ALL", "BLACK", "WHITE", "ASIAN", "NATAMER", "HISPANIC"]

        # Ensure the GeoDataFrame is in the correct CRS for basemaps (Web Mercator)
        gdf = gdf.to_crs(epsg=3857)

        # Group by Endpoint to ensure each endpoint is plotted separately
        grouped_endpoints = gdf.groupby("Endpoint")

        for endpoint, gdf_endpoint in grouped_endpoints:
            
            # Extract available race categories within this endpoint group
            available_races = [race for race in race_order if race in gdf_endpoint["Race"].unique()]

            # change colorbar label
            if "Mortality" in endpoint:
                colorbar_label = 'Annual Avoided number of deaths'
            elif "Asthma" in endpoint:
                colorbar_label = 'Annual Avoided number of people with the symptom'
            elif "Work" in endpoint:
                colorbar_label = 'Annual Avoided work day loss'

            if field == 'Mean_per_Pop':
                colorbar_label = colorbar_label + '\n (per million) '
        
            for race in available_races:

                # Filter data for the current race within the endpoint group
                gdf_race = gdf_endpoint[gdf_endpoint["Race"] == race]

                # Ensure the field exists
                if field not in gdf_race.columns:
                    #logger.info(f'Field {field} does not exist in the data for endpoint {endpoint}, race {race}. Skipping...')
                    continue

                # **Check if the field contains only zeros**
                if gdf_race[field].sum() == 0:
                    #logger.info(f'All values are zero for {endpoint}, {race}. Skipping...')
                    continue  # Skip this plot

                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot the spatial distribution
                vmin, vmax = gdf[field].min(), gdf[field].max()
                max_abs = max(abs(vmin), abs(vmax)) * 0.5

                if vmin >= 0 :
                    vmin = 0 
                    vmax = max_abs

                    # Define the number of discrete bins
                    num_bins = 9  # Adjust this number for more or fewer steps
                    color_map = plt.cm.Reds  # Choose a sequential colormap
                    bounds = np.linspace(vmin, vmax, num_bins + 1)  # Define color step boundaries
                    norm = mcolors.BoundaryNorm(bounds, color_map.N)  # Create a discrete colormap
                    
                else:
                    vmin = -max_abs
                    vmax = max_abs
                    # Define the number of discrete bins
                    num_bins = 18  # Adjust this number for more or fewer steps
                    color_map = plt.cm.get_cmap("bwr")  # Reverse coolwarm to get cool (blue) for negative and warm (red) for positive
                    bounds = np.linspace(vmin, vmax, num_bins + 1)  # Define color step boundaries
                    norm = mcolors.BoundaryNorm(bounds, color_map.N)  # Create a discrete colormap


                if region_name == 'Nation':
                    set_edgecolor= 'none'
                else: 
                    set_edgecolor='black'
                gdf_race.plot(column=field, cmap=color_map, norm = norm, vmin=vmin, vmax=vmax, 
                            legend=False, edgecolor=set_edgecolor, ax=ax, markersize=30, alpha=0.8)

                # Add a basemap
                ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=6)

                ax.set_title(f'{endpoint} - {race} ({region_name})')

                # Calculate summary statistics for this race
                total_target = gdf_race[field].sum()
                max_target = gdf_race[field].max()
                min_target = gdf_race[field].min()

                # Display summary stats on the plot
                ax.text(0.5, -0.15, f'Total: {total_target:.3f}\nMax: {max_target:.3f}\nMin: {min_target:.3f}', 
                        ha='center', va='center', transform=ax.transAxes, fontsize=12, color='black')


                # Add a color bar
                sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
                sm._A = []
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  
                fig.colorbar(sm, cax=cbar_ax).set_label(f'{colorbar_label}')

                # Adjust layout
                plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.2)

                # Save the figure for each endpoint-race combination
                output_filename = f'{field}_{endpoint}_{race}_{region_name}_with_basemap.png'
                output_path = os.path.join(output_dir, output_filename)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f'Saved: {output_path}')

    def modify_geojson(self, geojson_data, column):

        geojson_dict = json.loads(geojson_data)
        
        # Display properties to add to each feature
        display_properties = {
            'color': '#808080',
            'weight': 1,
            'fillOpacity': 0.75,
            'type': 'polygon',
            'iconFile': 'none',
            'layerOrder': 5,
            'onLoad': False,
            'legendEntry': 'none'
        }

        # Add displayProperties at the top level
        geojson_dict['displayProperties'] = display_properties

        if column == 'TotalPopD':
            geojson_dict['QuantityDescriptor'] = "Changes in total premature deaths"
        elif column == 'TotalPM25':
            geojson_dict['QuantityDescriptor'] = "Changes in surface PM2.5 concentrations"
        
        # Rename the quantity column to "Quantity"
        for feature in geojson_dict['features']:
            feature['properties']['Quantity'] = feature['properties'].pop(column)
        
        modified_geojson = json.dumps(geojson_dict, indent=2)
        modified_geojson = f"var INMAP_{column} = {modified_geojson};"
        
        return modified_geojson


    def save_benmap_json(self, gdf_diff, columns_to_save, webdata_path):

        for column in columns_to_save:

            gdf_column = gdf_diff[['geometry', column]].copy()
            
            if column == 'TotalPopD':
                threshold = 0
            elif column == 'TotalPM25':
                threshold = 0.0000001

            gdf_filtered = gdf_column[(gdf_column[column].abs() > threshold)]

            geojson_data = gdf_filtered.to_json()
            modified_geojson =self.modify_geojson(geojson_data, column)

            filename = os.path.join(webdata_path, f'INMAP_{column}.json')
            with open(filename, 'w') as f:
                f.write(modified_geojson)

            logger.info(f"GeoJSON data for column '{column}' has been saved to '{filename}'.")

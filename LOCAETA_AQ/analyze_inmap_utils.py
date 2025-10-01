import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import seaborn as sns
import folium
from folium import LayerControl
import branca.colormap as cm
import numpy as np
import json
import contextily as ctx
import pandas as pd
import logging
from .run_inmap_utils import INMAP_Processor

# Suppress all warnings in jupyter notebook
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class INMAP_Analyzer:
    """
    A class to analyze INMAP simulation results 
    """

    def __init__(self, cfg):
        """
        Initialize the processor with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with file paths and parameters
        """
        self.cfg = cfg


    def process_run_pair(self, run_name, paths, inmap_run_dir):

        base_path = os.path.join(inmap_run_dir, paths['base'])
        sens_path = os.path.join(inmap_run_dir, paths['sens'])

        gdf_base = self.load_shapefile(base_path)
        gdf_sens = self.load_shapefile(sens_path)

        logger.info(f"Processing {run_name} - Base: {base_path}, Sensitivity: {sens_path}")

        gdf_diff = gdf_base[['geometry']].copy()  # Copy only the geometry column
        columns = gdf_base.columns.difference(['geometry', 'FIPS'])

        # Compute the differences and add them as new fields to gdf_diff
        for field in columns:
            gdf_diff[field] = gdf_sens[field] - gdf_base[field]
            gdf_diff[field+'_base'] = gdf_base[field]  # save base run results

        return gdf_diff

    def determine_output_type(self, gdf_diff, INMAP_cols, SRmatrix_cols):
        if set(INMAP_cols).issubset(gdf_diff.columns):
            columns = INMAP_cols
            output_type = 'inmap_run'
            area_weight_list = ['NH3', 'NOx', 'SOA', 'SOx', 'PNH4', 'PNO3', 'PSO4', 'PrimPM25', 'TotalPM25']
        elif set(SRmatrix_cols).issubset(gdf_diff.columns):
            columns = SRmatrix_cols 
            output_type = 'source_receptor'
            area_weight_list = ['pNH4', 'pNO3', 'pSO4', 'PrimPM25', 'TotalPM25']
        else:
            raise ValueError("The columns in gdf_diff do not match expected set of columns.")
        
        return columns, output_type, area_weight_list


    def compute_and_print_summaries(self, gdf_diff, columns, area_weight_list, output_dir):
        # Compute column sums for health benefits
        column_sums = gdf_diff[columns].sum()
        logger.info(f"Column Sums:\n, {column_sums}")

        # Compute area-weighted averages for AQ benefits
        gdf_diff['area'] = gdf_diff.geometry.area
        area_weighted_averages = {}

        for field in area_weight_list:
            area_weighted_averages[field] = (gdf_diff[field] * gdf_diff['area']).sum() / gdf_diff['area'].sum()

        logger.info("Area-Weighted Averages [ug m-3]:")
        for key, value in area_weighted_averages.items():
            logger.info(f"{key}: {value}")

        # Save area-weighted averages to CSV
        output_file = os.path.join(output_dir, "area_weighted_averages.csv")
        df = pd.DataFrame(area_weighted_averages.items(), columns=["Species", "Area-Weighted Average"])
        df.to_csv(output_file, index=False)
        logger.info(f"Saved area-weighted averages to {output_file}")

        # Save column sums to CSV
        output_file = os.path.join(output_dir, "mortality_sums.csv")
        #df = pd.DataFrame(column_sums.items(), columns=["Species", "Area-Weighted Average"])
        column_sums.to_csv(output_file, index=True)
        logger.info(f"Saved column sums to {output_file}")

        return column_sums, area_weighted_averages

    def create_interactive_map(self, gdf_diff, field, output_dir):

        if gdf_diff.crs is None:
            gdf_diff.set_crs(epsg=4326, inplace=True)
        else:
            gdf_diff.to_crs(epsg=4326, inplace=True)

        # Create an index column for referencing in the choropleth map
        # gdf_diff = gdf_diff.reset_index()

        min_value = gdf_diff[field].min()
        max_value = gdf_diff[field].max()

        colormap = cm.LinearColormap(
            colors=['blue', 'white', 'red'],
            vmin=-0.1, #min_value, #-2, #0.5,
            vmax=0.1, #max_value, #2, #0.5,
            caption=f'{field} Difference'
        )

        # Create a base map centered around the centroid of the GeoDataFrame
        map_center = [gdf_diff.geometry.centroid.y.mean(), gdf_diff.geometry.centroid.x.mean()]
        m = folium.Map(location=map_center, zoom_start=6)

        folium.GeoJson(
            gdf_diff,
            name=field,
            style_function=lambda x: {
                'fillColor': colormap(x['properties'][field]),
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.7,
            },
            tooltip=folium.features.GeoJsonTooltip(
                fields=[field],
                aliases=[f'{field} Difference:'],
                localize=True
            )
        ).add_to(m)

        colormap.add_to(m)

        LayerControl().add_to(m)

        # Find the location of the maximum value
        max_idx = gdf_diff[field].idxmax()
        max_location = gdf_diff.loc[max_idx, 'geometry'].centroid

        # Add a circle marker at the location with the maximum value
        folium.CircleMarker(
            location=[max_location.y, max_location.x],
            radius=10,  # Adjust the radius as needed
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.6,
            popup=f'Max {field}: {max_value}'
        ).add_to(m)

        # Find the location of the minimum value
        min_idx = gdf_diff[field].idxmin()
        min_location = gdf_diff.loc[min_idx, 'geometry'].centroid

        # Add a circle marker at the location with the minimum value
        folium.CircleMarker(
            location=[min_location.y, min_location.x],
            radius=10,  # Adjust the radius as needed
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup=f'Min {field}: {min_value}'
        ).add_to(m)

        m.save(os.path.join(output_dir, f"{field}_interactive_map.html"))



    def barplot_health_aq_benefits(self, area_weighted_averages, column_sums, output_dir):
        """
        Generates two bar plots: 
        - Impact of CCS emissions on surface air quality
        - Impact of CCS emissions on total premature mortality
        """
        # Convert dictionaries to DataFrame for easier plotting
        df_aq = pd.DataFrame(list(area_weighted_averages.items()), columns=['Field', 'Value'])
        df_pop = pd.DataFrame({'Field': column_sums.index, 'Value': column_sums.values})

        # Ensure values are numeric (fixes missing annotations issue)
        df_aq['Value'] = pd.to_numeric(df_aq['Value'])
        df_pop['Value'] = pd.to_numeric(df_pop['Value'])

        # Set Seaborn style
        sns.set_style('whitegrid')

        def format_value(value):
            """Formats values for bar annotations."""
            if abs(value) < 0.01:
                return f'{value:.2e}'  # Scientific notation
            elif abs(value) < 1:
                return f'{value:.2f}'  # Two decimal places
            else:
                return f'{int(value)}'  # No decimal places

        def plot_bar(df, title, y_label, output_filename):
            """Helper function to create bar plots with improved aesthetics."""
            plt.figure(figsize=(14, 8))

            # Define color palette
            colors = ['blue' if val < 0 else 'red' for val in df['Value']]

            # Create bar plot
            ax = sns.barplot(data=df, x='Field', y='Value', palette=colors, errorbar=None, edgecolor='black')

            # Title and labels
            plt.title(title, fontsize=25, fontweight='bold', pad=20)
            plt.xlabel('Fields', fontsize=18)
            plt.ylabel(y_label, fontsize=18)
            plt.xticks(rotation=45, fontsize=18)
            plt.yticks(fontsize=16)
            plt.axhline(0, color='black', linewidth=1.2, linestyle='--')
            plt.grid(axis='y', linestyle='--', alpha=0.5)

            # Add value labels on bars
            for p in ax.patches:
                value = p.get_height()
                if not np.isnan(value):  # Skip NaNs and tiny values
                    text_y_pos = 0 # value + (abs(value)* 0.2* np.sign(value))
                    ax.annotate(
                        format_value(value),
                        (p.get_x() + p.get_width() / 2., text_y_pos),
                        ha='center', 
                        va='bottom',
                        fontsize=12,
                        fontweight='bold')

            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight')
            #plt.show()

        # Plot Air Quality Impact
        plot_bar(df_aq, 'Impact of CCS Emissions on Surface Air Quality',
                'Area-Weighted Average [ug/m³]', 'CCS_impact_on_area_weighted_AQ.png')

        # Plot Population Health Impact
        plot_bar(df_pop, 'Impact of CCS Emissions on Total Premature Mortality',
                'Total Premature Mortality by CCS', 'CCS_impact_on_total_deaths.png')

    def load_shapefile(self, shapefile_path):
        gdf = gpd.read_file(shapefile_path)
        if gdf.crs is None or gdf.crs.is_geographic:
            gdf = gdf.set_crs('EPSG:4269', allow_override=True)
        return gdf

    # Function to plot the percent change of each field and its "_base" version with a basemap
    def plot_spatial_distribution_percent_change_with_basemap(self, gdf, field, output_dir):
        
        # Ensure the GeoDataFrame is in the correct CRS for basemaps (Web Mercator)
        gdf = gdf.to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(10, 6))

        col_diff = f'{field}'
        col_base = f'{field}_base'

        # Ensure both the current and "_old" columns exist in the GeoDataFrame
        if col_diff not in gdf.columns or col_base not in gdf.columns:
            raise ValueError(f'Columns {col_diff} or {col_base} do not exist in the data.')
        
        # Calculate the percent change, avoiding division by zero
        gdf[ field+'_percent_change'] = (gdf[col_diff] / gdf[col_base]) * 100

        # Plot the spatial distribution of the percent change
        vmin, vmax = -1, 1 # -1, 1  # Fixed color scale from -50% to 50%

        gdf.plot(column=field+'_percent_change', cmap='coolwarm', vmin=vmin, vmax=vmax, legend=False, edgecolor='none', 
                    linewidth=0, ax=ax, markersize=30, alpha=0.8)  # Increase marker size and reduce transparency

        # Add a basemap (using OpenStreetMap)
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=6)

        ax.set_title(f'Percent Change in {field}')

        # Calculate the total, max, and min percent change
        total_diff = gdf[col_diff].sum()
        total_base = gdf[col_base].sum()
        total_percent_change = (total_diff / total_base) * 100
        max_percent_change = gdf[field+'_percent_change'].max()
        min_percent_change = gdf[field+'_percent_change'].min()

        # Display the total, max, and min percent changes on the plot
        ax.text(0.5, -0.15, f'Total Percent Change: {total_percent_change:.3f}%\nMax Percent Change: {max_percent_change:.3f}%\nMin Percent Change: {min_percent_change:.3f}%', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='black')

        # Add a color bar for the field
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax))
        sm._A = []
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position color bar to avoid overlap
        fig.colorbar(sm, cax=cbar_ax).set_label(f'{field} Percent Change')

        # Adjust layout to prevent overlap
        plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.2)

        # Save the figure for each field as a separate file
        plt.savefig(os.path.join(output_dir, f'{field}_Percent Change_with_basemap.png'), dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to avoid overlapping plots

    def subset_state(self, gdf, state_fips):

        # read county polygon information from this shapefile (needed for non-point sources)
        shapefile_path = self.cfg["nei_emissions"]['input']['county_shapefile_dir']
        gdf_fips = gpd.read_file(shapefile_path)

        # this is necessary for basemap plotting
        gdf_fips = gdf_fips.to_crs(epsg=3857)
        target_proj = gdf.crs

        if gdf_fips.crs != target_proj:
            logger.info(f"Reprojecting from {gdf_fips.crs} to {target_proj}")
            gdf_fips = gdf_fips.to_crs(target_proj)

        # Ensure state_fips is a list
        if not isinstance(state_fips, list):
            state_fips = [state_fips]
        
        # Collect all states of interest, including neighbors
        all_fips = set(state_fips)

        # Get all county geometries for the state and merge them into a single geometry
        state_geom = gdf_fips[gdf_fips['STATEFP'].isin(all_fips)].geometry.unary_union

        # Subset your dataset using spatial intersection
        gdf_co = gdf[gdf.intersects(state_geom)]

        # check new dataset
        #logger.info(gdf_co.head())

        return gdf_co

    def normalize_total_pop_by_area(self, gdf, pop_col="TotalPopD", area_col="area_km2"):
        """
        Normalize population by grid area.

        Parameters:
        - gdf: GeoDataFrame with population and geometry
        - pop_col: Column name for total population (default: "TotalPopD")
        - area_col: Optional column name to store the computed area in km² (default: "area_km2")

        Returns:
        - GeoDataFrame with a new column "TotalPopD_density" (population per km²)
        """
        # Ensure the GeoDataFrame has a projected CRS (for accurate area calculation)
        if gdf.crs is None or not gdf.crs.is_projected:
            raise ValueError("GeoDataFrame must have a projected CRS to calculate area accurately.")

        # Calculate area in square kilometers
        gdf[area_col] = gdf.geometry.area / 1e6  # from m² to km²

        # Normalize population by area
        gdf["TotalPopD_density"] = gdf[pop_col] / gdf[area_col]

        return gdf

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


    def save_inmap_json(self, gdf_diff, columns_to_save, webdata_path):

        for column in columns_to_save:

            gdf_column = gdf_diff[['geometry', column]].copy()
            
            if column == 'TotalPopD':
                threshold = 0
            elif column == 'TotalPM25':
                threshold = 0.0000001

            gdf_filtered = gdf_column[(gdf_column[column].abs() > threshold)]

            geojson_data = gdf_filtered.to_json()
            modified_geojson = self.modify_geojson(geojson_data, column)

            filename = os.path.join(webdata_path, f'INMAP_{column}.json')
            with open(filename, 'w') as f:
                f.write(modified_geojson)

            logger.info(f"GeoJSON data for column '{column}' has been saved to '{filename}'.")


    def compare_pm25_mortality_changes(self, gdf_diff,output_dir, run_name): 

        sign_match = np.sign(gdf_diff['TotalPM25']) == np.sign(gdf_diff['TotalPopD'])
        mismatch_percentage = (1 - sign_match.mean()) * 100
        logger.info(f"{mismatch_percentage:.2f}% of the observations have mismatched signs for TotalPM25 and TotalPopD.")

        magnitude_ratio = gdf_diff['TotalPM25'].abs() / gdf_diff['TotalPopD'].abs()  
        median_ratio = magnitude_ratio.median()
        logger.info(f"The median absolute magnitude ratio (TotalPM25 / TotalPopD) is: {median_ratio:.2f}")

        plt.figure(figsize=(10, 6))
        plt.scatter(gdf_diff['TotalPopD'], gdf_diff['TotalPM25'], alpha=0.5)
        plt.xlabel('TotalPopD')
        plt.ylabel('TotalPM25')
        plt.title(f'Scatter Plot of TotalPopD vs. TotalPM25 for {run_name}')
        plt.grid(True)
        # Add a diagonal line for reference
        plt.plot(plt.xlim(), plt.ylim(), ls="--", c=".3")
        plt.savefig(os.path.join(output_dir, 'INMAP_PM_mortality_scatter_plot.png'))
        plt.close() 

    def collect_analysis_infos(self, scenario, run_names):
        """
        Collect information for INMAP analysis.

        Parameters
        ----------
        scenario : str
            Scenario key (e.g., 'datacenter_emissions', 'ccs_emissions').
        run_names : list of str
            List of run names to process.

        Returns
        -------
        dict
            Dictionary keyed by run_name (or run_name_case) with structure:
            {
                run_name: {
                    'base': <base shapefile path>,
                    'sens': <sensitivity shapefile path>
                },
                ...
            }
        """
        run_infos = {}

        run_inmap_obj = INMAP_Processor(self.cfg)

        output_base = self.cfg['inmap']['output']['output_dir']

        for target_run_name in run_names:
            _, run_name = run_inmap_obj.get_emission_paths(scenario, target_run_name)
            sens_run_output = os.path.join(output_base, run_name)

            # Base emissions
            if self.cfg[scenario]['has_own_base_emission']:
                base_run_output = os.path.join(output_base, run_name+ "_base") 
            else:
                # if it doesnt have their own base emissions, use the default base run 
                base_run_output = os.path.join(output_base, self.cfg['inmap']['analyze']['default_base_run']) 

            run_infos[run_name] = {
                'base': os.path.join(base_run_output, "2020nei_output_run_steady.shp"),
                'sens': os.path.join(sens_run_output, "2020nei_output_run_steady.shp")
            }

            # Separate cases
            separate_cases = self.cfg.get('stages', {}).get('separate_case_per_each_run') or []
            for case_name in separate_cases:
                _, run_name_case = run_inmap_obj.get_emission_paths(scenario, f"{target_run_name}_{case_name}")
                sens_run_output_case = os.path.join(output_base, run_name_case)
                run_infos[run_name_case] = {
                    'base': os.path.join(base_run_output, "2020nei_output_run_steady.shp"),
                    'sens': os.path.join(sens_run_output_case, "2020nei_output_run_steady.shp")
                }

        # Exclude other runs if "run_only_separate_case" is true. 
        if self.cfg["stages"]["run_only_separate_case"]:
            separate_cases = self.cfg.get('stages', {}).get('separate_case_per_each_run') or []
            for run_name in list(run_infos.keys()):
                if not any(case_name in run_name for case_name in separate_cases):
                    del run_infos[run_name]

        return run_infos
    

    def setup_output_dirs(self, config):
        """Ensure output directories exist and return paths."""
        output_dir = config['output']['plots_dir']
        json_output_path = config['output']['json_dir']
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(json_output_path, exist_ok=True)
        return output_dir, json_output_path

    def process_one_run(self, run_name, paths, config,
                        inmap_columns, source_receptor_columns):
        """Process one run pair and return the diff dataframe and metadata."""
        inmap_run_dir = config['output']['output_dir']
        gdf_diff = self.process_run_pair(run_name, paths, inmap_run_dir)

        columns_list, output_type, area_weight_list = self.determine_output_type(
            gdf_diff, inmap_columns, source_receptor_columns
        )
        logger.info(f"Run {run_name}: data is from an {output_type} output.")

        return gdf_diff, output_type, columns_list, area_weight_list


    def handle_inmap_run(self, gdf_diff, run_output_dir, run_name):
        """Special handling for inmap_run outputs (cleanup, normalization, plots)."""
        # Compare PM2.5 and mortality changes
        self.compare_pm25_mortality_changes(gdf_diff, run_output_dir, run_name)

        # Identify and remove problematic rows
        to_check = gdf_diff[(abs(gdf_diff['TotalPopD_base']) > abs(gdf_diff['TotalPop_base']))]
        if not to_check.empty:
            logger.warning(f"{len(to_check)} rows removed due to inconsistent mortality in {run_name}")
            to_check.to_csv(os.path.join(run_output_dir, "wrong_mortality_rows.csv"), index=True)
            gdf_diff = gdf_diff.drop(to_check.index)

        # Normalize premature deaths by grid area
        gdf_diff = self.normalize_total_pop_by_area(gdf_diff, pop_col="TotalPopD", area_col="area_km2")

        return gdf_diff


    def plot_for_states(self, gdf_diff, variable, run_output_dir, state_regions):
        """Plot spatial distribution either for selected states or for the whole dataset."""
        if state_regions:
            for state, state_fips in state_regions.items():
                logger.info(f"Subsetting data for {state} ({state_fips}) in {variable}")
                gdf_subset = self.subset_state(gdf_diff, state_fips)
                self.plot_spatial_distribution_percent_change_with_basemap(
                    gdf_subset, variable, run_output_dir
                )
        else:
            self.plot_spatial_distribution_percent_change_with_basemap(
                gdf_diff, variable, run_output_dir
            )


    def save_json_outputs(self, gdf_diff, inmap_to_geojson, run_json_output_path, state_regions):
        """Save GeoJSON files for InMAP results."""
        if state_regions:
            # Save subset for the last state only (matches your current logic)
            for state, state_fips in state_regions.items():
                gdf_subset = self.subset_state(gdf_diff, state_fips)
                self.save_inmap_json(gdf_subset, inmap_to_geojson, run_json_output_path)
        else:
            self.save_inmap_json(gdf_diff, inmap_to_geojson, run_json_output_path)


##############################################################################
####### The functions below are for processing INMAP output for BenMAP #######
##############################################################################

    def calculate_grid_intersected_areas(self, inmap, gdf_grids):
        # Calculate the area of each grid cell
        inmap['grid_cell_area'] = inmap.geometry.area
        
        # Perform an intersection to calculate intersected areas
        inmap_with_fips = gpd.overlay(inmap, gdf_grids[['ID', 'geometry']], how='intersection')

        # Calculate the intersected area
        inmap_with_fips['intersected_area'] = inmap_with_fips.geometry.area
        
        return inmap_with_fips

    def reshape_inmap_to_grid_data(self,inmap_with_fips, gdf_grids):
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

    def save_AQ_csv(self, grid_level_gdf, output_csv_path):
        # Prepare data for CSV
        grid_level_gdf['Annual Metric'] = 'Mean'
        grid_level_gdf['Metric'] = 'D24HourMean'
        grid_level_gdf['Seasonal Metric'] = 'QuarterlyMean'
        inmap_csv = grid_level_gdf[['ROW', 'COL', 'Metric','Seasonal Metric','Annual Metric','area_weighted_avg_TotalPM25']]
        # Rename the columns name to BenMAP format
        inmap_csv = inmap_csv.rename(columns={'ROW': 'Row', 'COL':'Column','area_weighted_avg_TotalPM25':'Values'})
        
        # Save as CSV
        inmap_csv.to_csv(output_csv_path, index=False)

    def save_grid_shapefile(self, grid_level_gdf, output_shapefile_path):
        # Save as new shapefile
        grid_level_grid = grid_level_gdf[['ROW', 'COL', 'geometry','area_weighted_avg_TotalPM25']]
        grid_level_grid = grid_level_grid.rename(columns={'ROW': 'Row', 'COL':'Column'})
        grid_level_grid.to_file(output_shapefile_path)

    def process_inmap_to_benmap_inputs(self, inmap_output_path, grid_shapefile_path, output_shapefile_path, output_csv_path, grid_level):

        inmap = gpd.read_file(inmap_output_path)
        inmap = inmap[['geometry', 'TotalPM25']]
        
        gdf_grids = gpd.read_file(grid_shapefile_path)

        if grid_level == 'tracts':
            # Create a unique ID for tracts using ROW and COL
            gdf_grids['ID'] = gdf_grids['ROW'].astype(str) + '_' + gdf_grids['COL'].astype(str)
        elif grid_level == 'county':  
            gdf_grids = gdf_grids.rename(columns={'FIPS': 'ID'})
            
        logger.info(f"Original CRS inmap: {inmap.crs} and CRS gdf_grids: {gdf_grids.crs}")
        
        # BenMAP's preferred projection for area calculation
        projection = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=23, standard_parallels=(29.5, 45.5))
        
        gdf_grids =  gdf_grids.to_crs(projection) 
        inmap = inmap.to_crs(projection) 
        logger.info(f"after geographic conversion, CRS inmap: {inmap.crs} and CRS gdf_grids: {gdf_grids.crs}")
        
        # Compute the intersection area for area-weighted concentrations
        inmap_with_grids = self.calculate_grid_intersected_areas(inmap, gdf_grids)
        
        # Reshape INMAP grids to grid level
        grid_level_gdf = self.reshape_inmap_to_grid_data(inmap_with_grids, gdf_grids)
        
        # Save shapefile and CSV
        self.save_grid_shapefile(grid_level_gdf, output_shapefile_path)
        self.save_AQ_csv(grid_level_gdf, output_csv_path)
        
        logger.info(f"Shapefile and CSV for {inmap_output_path} have been saved successfully.")
        
        return inmap, grid_level_gdf

    def plot_pm25_original_and_reshaped_results(self, inmap, grid_level_gdf, output_prefix, output_dir, grid_level):


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


    def process_and_plot(self, input_path, grid_shapefile, output_csv_path, run_name, benmap_input_dir, grid_level):
        output_shapefile_path = f"{benmap_input_dir}{run_name}_{grid_level}_inmap_grid.shp"
        inmap, grid_level_gdf = self.process_inmap_to_benmap_inputs(
            input_path,
            grid_shapefile,
            output_shapefile_path,
            output_csv_path,
            grid_level=grid_level
        )
        self.plot_pm25_original_and_reshaped_results(inmap, grid_level_gdf, run_name, benmap_input_dir, grid_level)
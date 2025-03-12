# model_analysis.py

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import geopandas as gpd
import folium
from folium import LayerControl
import branca.colormap as cm
import numpy as np
import json


def process_run_pair(run_name, paths, inmap_run_dir):
    base_path = os.path.join(inmap_run_dir, paths['base'])
    sens_path = os.path.join(inmap_run_dir, paths['sens'])

    gdf_base = load_shapefile(base_path)
    gdf_sens = load_shapefile(sens_path)

    print(f"Processing {run_name} - Base: {base_path}, Sensitivity: {sens_path}")

    gdf_diff = gdf_base[['geometry']].copy()  # Copy only the geometry column
    columns = gdf_base.columns.difference(['geometry', 'FIPS'])

    # Compute the differences and add them as new fields to gdf_diff
    for field in columns:
        gdf_diff[field] = gdf_sens[field] - gdf_base[field]

    return gdf_diff

def determine_output_type(gdf_diff, INMAP_cols, SRmatrix_cols):
    if set(INMAP_cols).issubset(gdf_diff.columns):
        columns = INMAP_cols
        output_type = 'inmap_run'
        area_weight_list = ['NH3', 'NOx', 'SOA', 'SOx', 'PNH4', 'PNO3', 'PSO4', 'PrimPM25', 'TotalPM25']
    elif set(SRmatrix_cols).issubset(gdf_diff.columns):
        columns = SRmatrix_cols 
        output_type = 'source_receptor'
        area_weight_list = ['pNH4', 'pNO3', 'pSO4', 'PrimPM25', 'TotalPM25']
    else:
        raise ValueError("The columns in gdf_diff do not match either expected set of columns.")
    
    return columns, output_type, area_weight_list


def compute_and_print_summaries(gdf_diff, columns, area_weight_list):

    # Compute column sums for health benefits
    column_sums = gdf_diff[columns].sum()
    print("Column Sums:\n", column_sums)

    # Compute area-weighted averages for AQ benefits
    gdf_diff['area'] = gdf_diff.geometry.area
    area_weighted_averages = {}

    for field in area_weight_list:
        area_weighted_averages[field] = (gdf_diff[field] * gdf_diff['area']).sum() / gdf_diff['area'].sum()

    print("Area-Weighted Averages [ug m-3]:")
    for key, value in area_weighted_averages.items():
        print(f"{key}: {value}")

    return column_sums, area_weighted_averages

def create_interactive_map(gdf_diff, field, output_dir):

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

def barplot_health_aq_benefits (area_weighted_averages, column_sums, output_dir): 

    plt.figure(figsize=(12, 6))
    colors = ['blue' if val < 0 else 'red' for val in area_weighted_averages.values()]
    plt.bar(area_weighted_averages.keys(), area_weighted_averages.values(), color=colors)
    plt.title('Impact of CSS emissions on surface air quality', fontsize=14)
    plt.xlabel('Fields', fontsize=12)
    plt.ylabel('Area-Weighted Average [ug m-3]', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f'CCS_impact_on_area_weighted_AQ.png'), dpi=300, bbox_inches='tight')


    plt.figure(figsize=(12, 6))
    colors = ['blue' if val < 0 else 'red' for val in column_sums.values]
    plt.bar(column_sums.index, column_sums.values, color=colors)
    plt.title('Impact of CSS emissions on populations', fontsize=14)
    plt.xlabel('Fields', fontsize=12)
    plt.ylabel('Total premature mortality by CSS', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f'CCS_impact_on_total_deaths.png'), dpi=300, bbox_inches='tight')


def load_shapefile(shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is None or gdf.crs.is_geographic:
        gdf = gdf.set_crs('EPSG:4269', allow_override=True)
    return gdf

def plot_difference_map(gdf1, gdf2, field, output_dir, year):
# NOT USED
    if gdf1.crs != gdf2.crs:
        gdf2 = gdf2.to_crs(gdf1.crs)

    gdf1 = gdf1.rename(columns={field: f'{field}_1'})
    gdf2 = gdf2.rename(columns={field: f'{field}_2'})
    joined_gdf = gpd.sjoin(gdf1, gdf2, how="inner", op='intersects')
    joined_gdf['difference'] = joined_gdf[f'{field}_1'] - joined_gdf[f'{field}_2']

    color_map = mcolors.LinearSegmentedColormap.from_list(
        'custom_colormap',
        ['blue', 'white', 'red']
    )
    norm = mcolors.BoundaryNorm([-50, -20, -10, -5, -1, 1, 5, 10, 20, 50], color_map.N)


    fig, ax = plt.subplots(1, 1, figsize=(15, 10), subplot_kw={'projection': ccrs.LambertConformal()})

    gdf1.plot(ax=ax, facecolor='none', edgecolor='gray', linewidth=0.5, transform=ccrs.PlateCarree())

    joined_gdf.plot(column='difference', cmap=color_map, linewidth=0.0, ax=ax, edgecolor='none', norm=norm, legend=False, transform=ccrs.PlateCarree(), alpha=0.9)


    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, linestyle=':', edgecolor='gray')


    sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label('Difference in PM2.5 Concentration')


    ax.set_title(f'Difference in PM2.5 Concentration between Two Runs ({year})', fontsize=16)

    # Set the extent to cover the CONUS region
    ax.set_extent([-125, -66.5, 24, 50], crs=ccrs.PlateCarree())

    plt.savefig(os.path.join(output_dir, f'difference_pm25_map_{year}.png'), dpi=300, bbox_inches='tight')



def modify_geojson(geojson_data, column):

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


def save_inmap_json(gdf_diff, columns_to_save, webdata_path):

    for column in columns_to_save:

        gdf_column = gdf_diff[['geometry', column]].copy()
        
        if column == 'TotalPopD':
            threshold = 0
        elif column == 'TotalPM25':
            threshold = 0.0001

        gdf_filtered = gdf_column[(gdf_column[column].abs() > threshold)]

        geojson_data = gdf_filtered.to_json()
        modified_geojson = modify_geojson(geojson_data, column)

        filename = webdata_path + f'INMAP_{column}.json'
        with open(filename, 'w') as f:
            f.write(modified_geojson)

        print(f"GeoJSON data for column '{column}' has been saved to '{filename}'.")


def compare_pm25_mortality_changes(gdf_diff,output_dir, run_name): 

    sign_match = np.sign(gdf_diff['TotalPM25']) == np.sign(gdf_diff['TotalPopD'])
    mismatch_percentage = (1 - sign_match.mean()) * 100
    print(f"{mismatch_percentage:.2f}% of the observations have mismatched signs for TotalPM25 and TotalPopD.")

    magnitude_ratio = gdf_diff['TotalPM25'].abs() / gdf_diff['TotalPopD'].abs()  
    median_ratio = magnitude_ratio.median()
    print(f"The median absolute magnitude ratio (TotalPM25 / TotalPopD) is: {median_ratio:.2f}")

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
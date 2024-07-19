import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'LOCAETA_AQ'))
if package_path not in sys.path:
    sys.path.append(package_path)

import inmap_analysis


def main(inmap_run_dir, output_dir, webdata_path, run_pairs, inmap_to_geojson):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    inmap_columns = ['AsianD', 'BlackD', 'LatinoD', 'NativeD', 'WhitNoLatD', 'TotalPopD']
    source_receptor_columns = ['deathsK', 'deathsL']

    for run_name, paths in run_pairs.items():
        gdf_diff = inmap_analysis.process_run_pair(run_name, paths, inmap_run_dir)

        columns_list, output_type, area_weight_list = inmap_analysis.determine_output_type(gdf_diff,inmap_columns,source_receptor_columns )
        print(f"The data is from an {output_type} output.")

        # Remove the row with the minimum TotalPopD for inmap_run
        if output_type == 'inmap_run':

            # compare the changes of PopD matches PM25.
            inmap_analysis.compare_pm25_mortality_changes(gdf_diff,output_dir, run_name)

            #min_pop_idx = gdf_diff['TotalPopD'].idxmin()
            #gdf_diff = gdf_diff.drop(index=min_pop_idx)
            
            for v in inmap_to_geojson:
                inmap_analysis.create_interactive_map(gdf_diff, v, output_dir, run_name)

        # Compute summaries and print them
        column_sums, area_weighted_averages = inmap_analysis.compute_and_print_summaries(gdf_diff, columns_list, area_weight_list)

        # Create a barplot of total premature deaths and area-weighted AQ
        inmap_analysis.barplot_health_aq_benefits(area_weighted_averages, column_sums, output_dir, run_name)

        # Create geojson files for columns_to_save
        inmap_analysis.save_inmap_json(gdf_diff, inmap_to_geojson, webdata_path, run_name)


if __name__ == "__main__":

    inmap_run_dir = '/Users/yunhalee/Documents/LOCAETA/RCM/INMAP/inmap-1.9.6/outputs/'
    analysis_output_dir = '/Users/yunhalee/Documents/LOCAETA/LOCAETA_AQ/outputs/model_analysis/'
    webdata_path = '/Users/yunhalee/Documents/LOCAETA/github/LOCAETA/WebTool/Data/'

    # Define pairs of base and sensitivity runs
    run_pairs = {
        'LA_only': {
            'base': 'LA_point_CSS/2020nei_LA_point_CSS_output_run_steady.shp',
            'sens': 'LA_point_CSS_reduced_emis/2020nei_LA_point_CSS_reduced_emis_output_run_steady.shp'
        },
        'CONUS': {
            'base': 'nei2020/2020nei_output_run_steady.shp',
            'sens': 'nei2020_LA_CCS/2020nei_output_run_steady.shp'
        }
    }

    inmap_to_geojson = ['TotalPopD', 'TotalPM25']

    main(inmap_run_dir, analysis_output_dir, webdata_path, run_pairs, inmap_to_geojson)

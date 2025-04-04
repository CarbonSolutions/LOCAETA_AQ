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
    state_regions = {"LA": ['22','05', "28", "48"]}   # {"CO": '08'} 

    for run_name, paths in run_pairs.items():
        gdf_diff = inmap_analysis.process_run_pair(run_name, paths, inmap_run_dir)

        columns_list, output_type, area_weight_list = inmap_analysis.determine_output_type(gdf_diff,inmap_columns,source_receptor_columns)
        print(f"The data is from an {output_type} output.")


        # create a directory for each run pair
        run_output_dir = os.path.join(output_dir, run_name)
        if not os.path.exists(run_output_dir):
            os.makedirs(run_output_dir)

        run_webdata_path = os.path.join(webdata_path, run_name)
        if not os.path.exists(run_webdata_path):
            os.makedirs(run_webdata_path)

        # Remove the row with the minimum TotalPopD for inmap_run
        if output_type == 'inmap_run':

            # compare the changes of PopD matches PM25.
            inmap_analysis.compare_pm25_mortality_changes(gdf_diff,run_output_dir, run_name)

            # Somehow one grid has larger mortality change than population..
            to_check = gdf_diff[(abs(gdf_diff['TotalPopD_base']) > abs(gdf_diff['TotalPop_base']))]
            print("Rows to be deleted due to wrong mortality:\n", to_check)
            to_check.to_csv( run_output_dir + "/wrong_mortality_rows.csv", index=True)

            gdf_diff = gdf_diff.drop(to_check.index)

            for v in inmap_to_geojson:
                inmap_analysis.create_interactive_map(gdf_diff, v, run_output_dir)

                ## This plot must only for the state (otherwise it takes very long to plot the map)
                for state, state_fips in state_regions.items():
                    print(f"subsetting the dataset for the state {state_fips}")
                    gdf_subset = inmap_analysis.subset_state(gdf_diff, state_fips)
                    inmap_analysis.plot_spatial_distribution_percent_change_with_basemap(gdf_subset, v, run_output_dir)

        # Compute summaries and print them
        column_sums, area_weighted_averages = inmap_analysis.compute_and_print_summaries(gdf_diff, columns_list, area_weight_list, run_output_dir)

        # Create a barplot of total premature deaths and area-weighted AQ
        inmap_analysis.barplot_health_aq_benefits(area_weighted_averages, column_sums, run_output_dir)

        # Create geojson files for columns_to_save
        inmap_analysis.save_inmap_json(gdf_subset, inmap_to_geojson, run_webdata_path)


if __name__ == "__main__":

    inmap_run_dir = '/Users/yunhalee/Documents/LOCAETA/RCM/INMAP/inmap-1.9.6-gridsplit/outputs/'
    analysis_output_dir = '/Users/yunhalee/Documents/LOCAETA/LOCAETA_AQ/outputs/model_analysis/'
    webdata_path = '/Users/yunhalee/Documents/LOCAETA/github/LOCAETA/WebTool/Data/'

    # Define pairs of base and sensitivity runs
    run_pairs = {
        # 'LA_CCS': {
        #      'base': 'base_nei2020/2020nei_output_run_steady.shp',
        #      'sens': 'LA_CCS/2020nei_output_run_steady.shp'
        #  },
        'LA_CCS_noNH3': {
             'base': 'base_nei2020/2020nei_output_run_steady.shp',
             'sens': 'LA_CCS_noNH3/2020nei_output_run_steady.shp'
         }

        #  'CO_CCS': {
        #      'base': 'base_nei2020/2020nei_output_run_steady.shp',
        #      'sens': 'CO_CCS/2020nei_output_run_steady.shp'
        #  },
        # 'CO_CCS_wo_NH3_VOC': {
        #      'base': 'base_nei2020/2020nei_output_run_steady.shp',
        #      'sens': 'CO_CCS_wo_NH3_VOC/2020nei_output_run_steady.shp'
        #  },
        # 'CO_Suncor_CCS_wo_NH3_VOC': {
        #     'base': 'base_nei2020/2020nei_output_run_steady.shp',
        #     'sens': 'CO_Suncor_CCS_wo_NH3_VOC/2020nei_output_run_steady.shp'
        # },
        # 'CO_Cherokee_CCS_wo_NH3_VOC': {
        #    'base': 'base_nei2020/2020nei_output_run_steady.shp',
        #    'sens': 'CO_Cherokee_CCS_wo_NH3_VOC/2020nei_output_run_steady.shp'
        # },
        # 'NEI_no_Landfill_2001411':{
        #     'base': 'base_nei2020/2020nei_output_run_steady.shp',
        #     'sens': 'NEI_no_Landfill_2001411/2020nei_output_run_steady.shp'}
    }
    inmap_to_geojson = ['TotalPopD', 'TotalPM25']

    main(inmap_run_dir, analysis_output_dir, webdata_path, run_pairs, inmap_to_geojson)

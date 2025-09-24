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


def main(inmap_run_dir, output_dir, webdata_path, run_pairs, inmap_to_geojson, state_regions):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    inmap_columns = ['AsianD', 'BlackD', 'LatinoD', 'NativeD', 'WhitNoLatD', 'TotalPopD']
    source_receptor_columns = ['deathsK', 'deathsL']

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

            # Normalize premature deaths by grid area
            gdf_diff = inmap_analysis.normalize_total_pop_by_area(gdf_diff, pop_col="TotalPopD", area_col="area_km2")

            for v in inmap_to_geojson:
                inmap_analysis.create_interactive_map(gdf_diff, v, run_output_dir)

                if v != 'TotalPopD_density': 
                    ## This plot must only for the state (otherwise it takes very long to plot the map)
                    if state_regions:
                        for state, state_fips in state_regions.items():
                            print(f"subsetting the dataset for the state {state_fips}")
                            gdf_subset = inmap_analysis.subset_state(gdf_diff, state_fips)
                            inmap_analysis.plot_spatial_distribution_percent_change_with_basemap(gdf_subset, v, run_output_dir)
                    else:
                        inmap_analysis.plot_spatial_distribution_percent_change_with_basemap(gdf_diff, v, run_output_dir)

        # Compute summaries and print them
        column_sums, area_weighted_averages = inmap_analysis.compute_and_print_summaries(gdf_diff, columns_list, area_weight_list, run_output_dir)

        # Create a barplot of total premature deaths and area-weighted AQ
        inmap_analysis.barplot_health_aq_benefits(area_weighted_averages, column_sums, run_output_dir)

        # Create geojson files for columns_to_save
        if state_regions:
            inmap_analysis.save_inmap_json(gdf_subset, inmap_to_geojson, run_webdata_path)
        else:
            inmap_analysis.save_inmap_json(gdf_diff, inmap_to_geojson, run_webdata_path)


if __name__ == "__main__":

    inmap_run_dir = '/Users/yunhalee/Documents/LOCAETA/RCM/INMAP/inmap-1.9.6-gridsplit/outputs/'
    analysis_output_dir = '/Users/yunhalee/Documents/LOCAETA/LOCAETA_AQ/outputs/model_analysis/'
    webdata_path = '/Users/yunhalee/Documents/LOCAETA/github/LOCAETA/WebTool/Data/'
    state_regions = {} #  {'TN':['47', '28', '05'] } # {"CO": '08'}  # {"LA": ['22','05', "28", "48"]}  #{} # #   #   # {"CO": '08'} 

    # Define pairs of base and sensitivity runs
    run_pairs = {
        # 'USA_zero_out_CCS': {
        #      'base': 'base_nei2020_jun2025/2020nei_output_run_steady.shp',
        #      'sens': 'USA_zero_out_CCS/2020nei_output_run_steady.shp'
        #  },
        # 'CO_CCS': {
        #      'base': 'base_nei2020_jun2025/2020nei_output_run_steady.shp',
        #      'sens': 'LA_CCS/2020nei_output_run_steady.shp'
        #  },
        # 'CO_CCS_wo_NH3_VOC': {
        #      'base': 'base_nei2020_jun2025/2020nei_output_run_steady.shp',
        #      'sens': 'LA_CCS_wo_NH3_VOC/2020nei_output_run_steady.shp'
        #  },
        # 'current_2020':{
        #     'base': 'current_2020_base/2020nei_output_run_steady.shp',
        #     'sens':'current_2020/2020nei_output_run_steady.shp'
        # },
        # 'current_easyhard':{
        #     'base': 'current_easyhard_base/2020nei_output_run_steady.shp',
        #     'sens':'current_easyhard/2020nei_output_run_steady.shp'
        # },
        # '2050_easyhard_decarb95':{
        #     'base': '2050_easyhard_decarb95_base/2020nei_output_run_steady.shp',
        #     'sens':'2050_easyhard_decarb95/2020nei_output_run_steady.shp'
        # },
        # '2050_easyhard_noIRA_111D':{
        #     'base': '2050_easyhard_noIRA_111D_base/2020nei_output_run_steady.shp',
        #     'sens':'2050_easyhard_noIRA_111D/2020nei_output_run_steady.shp'
        # }, 
            'current_easyhard_Food_Agr':{
            'base': 'current_easyhard_Food_Agr_base/2020nei_output_run_steady.shp',
            'sens':'current_easyhard_Food_Agr/2020nei_output_run_steady.shp'
        },
        '2050_easyhard_decarb95_Food_Agr':{
            'base': '2050_easyhard_decarb95_Food_Agr_base/2020nei_output_run_steady.shp',
            'sens':'2050_easyhard_decarb95_Food_Agr/2020nei_output_run_steady.shp'
        },
        '2050_easyhard_noIRA_111D_Food_Agr':{
            'base': '2050_easyhard_noIRA_111D_Food_Agr_base/2020nei_output_run_steady.shp',
            'sens':'2050_easyhard_noIRA_111D_Food_Agr/2020nei_output_run_steady.shp'
        },     
        # 'current_2020':{
        #     'base': 'current_2020_base/2020nei_output_run_steady.shp',
        #     'sens':'current_2020/2020nei_output_run_steady.shp'
        # },
        # 'current_2020_NorthernGrid_West':{
        #     'base': 'current_2020_base/2020nei_output_run_steady.shp',
        #     'sens':'current_2020_NorthernGrid_West/2020nei_output_run_steady.shp'
        # },
        # 'current_2020_SPP_North':{
        #     'base': 'current_2020_base/2020nei_output_run_steady.shp',
        #     'sens':'current_2020_SPP_North/2020nei_output_run_steady.shp'
        # },
        # 'current_2020_MISO_South':{
        #     'base': 'current_2020_base/2020nei_output_run_steady.shp',
        #     'sens':'current_2020_MISO_South/2020nei_output_run_steady.shp'
        # },
        # 'current_2020_PJM_East':{
        #     'base': 'current_2020_base/2020nei_output_run_steady.shp',
        #     'sens':'current_2020_PJM_East/2020nei_output_run_steady.shp'
        # },
        # 'current_2020_CAISO':{
        #     'base': 'current_2020_base/2020nei_output_run_steady.shp',
        #     'sens':'current_2020_CAISO/2020nei_output_run_steady.shp'
        # },
        # 'current_2020_MISO_Central':{
        #     'base': 'current_2020_base/2020nei_output_run_steady.shp',
        #     'sens':'current_2020_MISO_Central/2020nei_output_run_steady.shp'
        # },
        # 'decarb95_2050':{
        #     'base':'decarb95_2050_base/2020nei_output_run_steady.shp',
        #     'sens':'decarb95_2050/2020nei_output_run_steady.shp'
        # },
        # 'highREcost_2050':{
        #     'base':'highREcost_2050_base/2020nei_output_run_steady.shp',
        #     'sens':'highREcost_2050/2020nei_output_run_steady.shp'
        # },
        # 'highREcost_2050_base':{
        #     'base':'current_2020_base/2020nei_output_run_steady.shp',
        #     'sens':'highREcost_2050_base/2020nei_output_run_steady.shp'
        # },
        # 'decarb95_2050_base':{
        #     'base':'current_2020_base/2020nei_output_run_steady.shp',
        #     'sens':'decarb95_2050_base/2020nei_output_run_steady.shp'
        # },
        # 'TN_DataCenter_NOx_2ppm_clean_CCS': {
        #      'base': 'base_nei2020_jun2025/2020nei_output_run_steady.shp',
        #      'sens': 'Data_Center_NOx_2ppm_clean_CCS/2020nei_output_run_steady.shp'
        #  },
        # 'TN_DataCenter_NOx_2ppm': {
        #      'base': 'base_nei2020/2020nei_output_run_steady.shp',
        #      'sens': 'TN_DataCenter_NOx_2ppm/2020nei_output_run_steady.shp'
        #  }
        #  'CO_CCS': {
        #      'base': 'base_nei2020/2020nei_output_run_steady.shp',
        #      'sens': 'CO_CCS/2020nei_output_run_steady.shp'
        #  },
        # 'CO_CCS_wo_NH3_VOC': {
        #      'base': 'base_nei2020/2020nei_output_run_steady.shp',
        #      'sens': 'CO_CCS_wo_NH3_VOC/2020nei_output_run_steady.shp'
        #  },
        # 'NEI_no_Landfill_2001411': {
        #      'base': 'base_nei2020/2020nei_output_run_steady.shp',
        #      'sens': 'NEI_no_Landfill_2001411/2020nei_output_run_steady.shp'
        #  },
        # 'USA_CCS': {
        #      'base': 'base_nei2020_jun2025/2020nei_output_run_steady.shp',
        #      'sens': 'USA_CCS/2020nei_output_run_steady.shp'
        #  },
        # 'USA_CCS_wo_NH3_VOC': {
        #      'base': 'base_nei2020_jun2025/2020nei_output_run_steady.shp',
        #      'sens': 'USA_CCS_wo_NH3_VOC/2020nei_output_run_steady.shp'
        #  },
        # 'CO_Suncor_CCS_wo_NH3_VOC': {
        #     'base': 'base_nei2020/2020nei_output_run_steady.shp',
        #     'sens': 'CO_Suncor_CCS_wo_NH3_VOC/2020nei_output_run_steady.shp'
        # },
        # 'CO_Cherokee_CCS_wo_NH3_VOC': {
        #    'base': 'base_nei2020/2020nei_output_run_steady.shp',
        #    'sens': 'CO_Cherokee_CCS_wo_NH3_VOC/2020nei_output_run_steady.shp'
        # },
        # 'CO_landfills_scenario1':{
        #     'base': 'base_nei2020_jun2025/2020nei_output_run_steady.shp',
        #     'sens': 'CO_landfills_scenario1/2020nei_output_run_steady.shp'},
        # 'CO_landfills_scenario2':{
        #     'base': 'base_nei2020_jun2025/2020nei_output_run_steady.shp',
        #     'sens': 'CO_landfills_scenario2/2020nei_output_run_steady.shp'}
    }
    inmap_to_geojson = ['TotalPopD', 'TotalPM25', 'TotalPopD_density']

    main(inmap_run_dir, analysis_output_dir, webdata_path, run_pairs, inmap_to_geojson, state_regions)

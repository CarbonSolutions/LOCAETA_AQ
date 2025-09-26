import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from LOCAETA_AQ.analyze_inmap_utils import INMAP_Analyzer 
from LOCAETA_AQ.config_utils import load_config
import logging

# logging from run_workflow 
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def main(cfg):

    # Initialize processor
    processor = INMAP_Analyzer(cfg)

    config = cfg["inmap"]

    scenario = cfg['stages']['scenario']
    run_names = cfg['stages']['run_names']
    logger.info(f"Analyzing INMAP outputs for {scenario}: {run_names}")

    # Setup directories
    output_dir, json_output_dir = processor.setup_output_dirs(config)
    
    # Optional state filtering for mapping
    state_regions = config.get('analyze', {}).get('map_only_these_state_regions') or {}

    # Collect run info
    run_pairs = processor.collect_analysis_infos(scenario, run_names)

    # Constants for INMAP variables
    inmap_to_geojson = ['TotalPopD', 'TotalPM25', 'TotalPopD_density']
    inmap_columns = ['AsianD', 'BlackD', 'LatinoD', 'NativeD', 'WhitNoLatD', 'TotalPopD']
    source_receptor_columns = ['deathsK', 'deathsL']

    for run_name, paths in run_pairs.items():

        logger.info(f"starting {run_name}, {paths}")
        gdf_diff, output_type, columns_list, area_weight_list = processor.process_one_run(
            run_name, paths, config,
            inmap_columns, source_receptor_columns
        )

        logger.info(f"The data is from an {output_type} output.")

        # Create per-run output dirs
        run_output_dir = os.path.join(output_dir, run_name)
        run_json_output_path = os.path.join(json_output_dir, run_name)
        os.makedirs(run_output_dir, exist_ok=True)
        os.makedirs(run_json_output_path, exist_ok=True)

        # Handle special inmap_run processing
        if output_type == 'inmap_run':
            gdf_diff = processor.handle_inmap_run(gdf_diff, run_output_dir, run_name)

            for v in inmap_to_geojson:
                processor.create_interactive_map(gdf_diff, v, run_output_dir)
                if v != 'TotalPopD_density':
                    processor.plot_for_states( gdf_diff, v, run_output_dir, state_regions)

        # Summaries + plots
        column_sums, area_weighted_averages = processor.compute_and_print_summaries(
            gdf_diff, columns_list, area_weight_list, run_output_dir
        )
        processor.barplot_health_aq_benefits(area_weighted_averages, column_sums, run_output_dir)

        # Save GeoJSON
        processor.save_json_outputs(gdf_diff, inmap_to_geojson, run_json_output_path, state_regions)


    ######################################
    # convert INMAP output to BenMAP input
    ######################################
    benmap_AQ_input_dir = cfg["benmap"]["input"]["input_dir"]

    # default setup for benmap 
    grid_shapefile_path = cfg["benmap"]['default_setup']['shapefile_dir'] # '/Users/yunhalee/Documents/LOCAETA/RCM/BenMAP/grids/US Census Tracts/US Census Tracts.shp'
    grid_level = cfg["benmap"]['default_setup']['grid_level'] # 'county' # or 'tracts' 
    target_year = cfg["benmap"]['default_setup']['target_year'] # '2020'

    # Get all unique base runs from run_pairs
    unique_base_paths = {}
    for run_name, paths in run_pairs.items():
        base_path = paths["base"]
        if base_path not in unique_base_paths.values():  # only keep first occurrence
            unique_base_paths[run_name] = base_path
    
    logger.info(unique_base_paths)
    # --- Handle base runs ---
    if not cfg[scenario]['has_own_base_emission']:
        default_base_run = cfg['inmap']['analyze']['default_base_run']

        if len(unique_base_paths) == 1 :
            run_name, base_path = next(iter(unique_base_paths.items()))
            if default_base_run in base_path:
                logger.info(f"default base run is used: {base_path}")
                base_output_csv_path = f"{benmap_AQ_input_dir}/{default_base_run}_{grid_level}_inmap_{target_year}_pm25.csv"
                processor.process_and_plot(base_path, grid_shapefile_path, base_output_csv_path, run_name, benmap_AQ_input_dir, grid_level)
            else:
                raise ValueError(f"The file is NOT an expected base run: {base_path}")
        else:
            raise ValueError(f"When the default_base_run is used, there must be exactly 1 unique base: {unique_base_paths}")
    else:
        logger.info(f"multiple base run are used")
        for run_name, base_path in unique_base_paths.items():
            base_run_name = os.path.basename(os.path.dirname(base_path))
            logger.info(f"Checking base_run, {base_run_name}")

            if "base" in base_run_name:
                base_output_csv_path = f"{benmap_AQ_input_dir}/{base_run_name}_{grid_level}_inmap_{target_year}_pm25.csv"
                processor.process_and_plot(base_path, grid_shapefile_path, base_output_csv_path, run_name, benmap_AQ_input_dir, grid_level)
            else:
                raise FileExistsError(f"The file name is NOT an expected base run: {base_run_name}" )

    # --- Handle sensitivity runs (always per run_name) ---
    for run_name, paths in run_pairs.items():
        sens_path = paths["sens"]
        logger.info(f"Checking sens_run, {sens_path}")
        sens_output_csv_path = f"{benmap_AQ_input_dir}/control_{run_name}_{grid_level}_inmap_{target_year}_pm25.csv"
        processor.process_and_plot(sens_path, grid_shapefile_path, sens_output_csv_path, run_name, benmap_AQ_input_dir, grid_level)

if __name__ == "__main__":

    import logging
    import yaml
    from datetime import datetime

    # start logger 
    logfile = f"log_files/analyze_inmap_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,  # or DEBUG
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
           # logging.StreamHandler(),
            logging.FileHandler(logfile, mode="w")
        ]
    )
    
    cfg = load_config("config.yaml")
    main(cfg)



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
    print(state_regions)

    # Collect run info
    run_pairs = processor.collect_analysis_infos(scenario, run_names)

    # Constants for INMAP variables
    inmap_to_geojson = ['TotalPopD', 'TotalPM25', 'TotalPopD_density']
    inmap_columns = ['AsianD', 'BlackD', 'LatinoD', 'NativeD', 'WhitNoLatD', 'TotalPopD']
    source_receptor_columns = ['deathsK', 'deathsL']

    for run_name, paths in run_pairs.items():

        gdf_diff, output_type, columns_list, area_weight_list = processor.process_one_run(
            run_name, paths, config,
            inmap_columns, source_receptor_columns
        )

        print(f"The data is from an {output_type} output.")

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


if __name__ == "__main__":

    import logging
    import yaml
    from datetime import datetime

    # start logger 
    logfile = f"analyze_inmap_{datetime.now():%Y%m%d_%H%M%S}.log"
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



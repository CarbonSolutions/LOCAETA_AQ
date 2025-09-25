import warnings
import geopandas as gpd
import pandas as pd
from LOCAETA_AQ.analyze_benmap_utils import Benmap_Analyzer
from LOCAETA_AQ.run_benmap_utils import Benmap_Processor 
from LOCAETA_AQ.config_utils import load_config
import logging
import os

# logging from run_workflow 
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def main(cfg):

    # Initialize processor
    processor = Benmap_Analyzer(cfg)
    run_benmap_processor = Benmap_Processor(cfg)

    config = cfg["benmap"]
    scenario = cfg['stages']['scenario']
    run_names = cfg['stages']['run_names']

    logger.info(f"Analyzing INMAP outputs for {scenario}: {run_names}")
    
    # Optional state filtering for mapping
    state_regions = config.get('analyze', {}).get('map_only_these_state_regions') or {}

    # Benmap configuration
    benmap_root = config['input']['batchmode_dir']
    benmap_output_dir = config['output']['plots_dir']
    grid_shapefile_path = cfg["benmap"]['default_setup']['shapefile_dir'] 
    grid_level =  config['default_setup']['grid_level'] 
    target_year = config['default_setup']['target_year'] 

    benmap_output_type =['incidence' , 'valuation'] 

    # Collect run info
    output_pairs = run_benmap_processor.build_output_pairs(scenario, run_names, grid_level, target_year)
    logger.info(f"starting {output_pairs}")
        

    # read Benmap grid shapefile
    grid_gdf = gpd.read_file(grid_shapefile_path)
    grid_gdf.rename( columns={"COL":"Col", "ROW": "Row"}, inplace=True)

    # Define regions
    regions = {"Nation": None}
    if state_regions:
        regions.update(state_regions)

    for run_name, paths in output_pairs.items():
        control_run = paths["sens"]
        base_run = paths["base"]

        logger.info(f"Starting BenMAP analysis for: {run_name}")
        logger.info(f"  Base run: {base_run}")
        logger.info(f"  Control run: {control_run}")

        for benmap_output in benmap_output_type:

            benmap_output_file = f'{benmap_root}/APVR/{control_run}-{benmap_output}.csv'
            output_dir = f"{benmap_output_dir}/{run_name}"
            os.makedirs(output_dir, exist_ok=True)

            # STEP 1: Process BenMAP output and merge with grid
            final_df = processor.process_benmap_output(benmap_output_file, grid_gdf, benmap_output)

            # STEP 2: Analyze and plot by region
            for region_name, state_fips in regions.items():
                logger.info(f"Processing region: {region_name}")

                # Subset the final DataFrame based on the chosen state or national
                final_df_subset = processor.subset_data(final_df, state_fips)

                logger.info(f"Processing data for: {region_name}")

                # === Aggregate ===
                if benmap_output == 'incidence':
                    # Group by ['Endpoint', 'Pollutant', 'Author', 'Race'] and calculate the sum of 'Mean'
                    race_grouped_sum = final_df_subset.groupby(['Endpoint', 'Pollutant', 'Author', 'Race']).agg({'Mean': 'sum', "Population":"sum"}).reset_index()
                    race_grouped_sum['Mean_per_Pop'] = race_grouped_sum['Mean'] / race_grouped_sum['Population'] * 1000000  # Scale by 1,000,000 for readability

                    # Creating a table to show the values in the barplots
                    table_columns = ['Endpoint', 'Race', 'Mean', 'Mean_per_Pop']

                else:
                    # Group by ['Endpoint', 'Pollutant', 'Author', 'Race'] and calculate the sum of 'Mean'
                    race_grouped_sum = final_df_subset.groupby(['Endpoint', 'Pollutant', 'Author', 'Race']).agg({'Mean': 'sum'}).reset_index()

                    # Creating a table to show the values in the barplots
                    table_columns = ['Endpoint', 'Race', 'Mean']

                # Save summary table
                processor.create_csv(race_grouped_sum, table_columns, f'Summary Table Health Benefits by Race in {region_name}', output_dir)

                # STEP 3: Create spatial plots ===
                grouped = final_df_subset.groupby(['Endpoint', 'Pollutant', 'Author'])

                for (endpoint, pollutant, author), group in grouped:
                    # Due to the long running time, only map for mortality 
                    if "Mortality" in endpoint:
                        logger.info(f"Creating mortality maps for {endpoint} ({benmap_output})")

                        # Mean plot
                        processor.plot_spatial_distribution_benmap_with_basemap(group, "Mean", output_dir, region_name)
                
                    if benmap_output == 'incidence':
                        group['Mean_per_Pop'] = group.apply(lambda row: row['Mean'] / row['Population'] * 1000000, axis=1)  # Scale by 1,000,000 for readability
                        if "Mortality" in endpoint:
                            processor.plot_spatial_distribution_benmap_with_basemap(group, "Mean_per_Pop", output_dir, region_name)

    logger.info("BenMAP analysis completed successfully.")

if __name__ == "__main__":

    import logging
    import yaml
    from datetime import datetime

    # start logger 
    logfile = f"log_files/analyze_benmap_{datetime.now():%Y%m%d_%H%M%S}.log"
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

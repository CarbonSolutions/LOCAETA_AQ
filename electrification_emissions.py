import yaml
import geopandas as gpd
import os
import warnings
from LOCAETA_AQ.electrification_emissions_utils import ElectrificationEmissionProcessor
from LOCAETA_AQ.config_utils import load_config
import logging

# logging from run_workflow 
logger = logging.getLogger(__name__)

# Suppress all warnings
warnings.filterwarnings('ignore')

def main(cfg):

    config = cfg['electrification_emissions']
    # Add combined NEI file from cfg
    config['combined_nei_file'] = os.path.join(cfg['base_dirs']['nei_output_root'],
                                                cfg['nei_emissions']['output']['combined_pt_source_file'])

    overall_scenario = config['master_scenario']
    scenario_list = config['target_scenario']

    # Create output directories if not exists
    os.makedirs(os.path.join(config['output']['output_dir'], config['master_scenario']), exist_ok=True)
    os.makedirs(os.path.join(config['output']['plots_dir']), exist_ok=True)

    # Initialize processor
    processor = ElectrificationEmissionProcessor(config)

    # Read the point source emissions
    nei_all_pt = gpd.read_file(config['combined_nei_file'])

    # Reset index to ensure proper comparison
    nei_all_pt.reset_index(drop=True, inplace=True)

    col_dict = {}
    for poll in processor.NEI_cols:
        col_dict[poll] = f'{poll}_nei'

    nei_all_pt.rename(columns = col_dict, inplace=True)

    nei_all_pt.head()

    # Build dictionary mapping scenario name -> emission file name
    if overall_scenario != "Full_USA":
        scen_emis_list =  {scen: f"{scen}_{overall_scenario}" for scen in scenario_list}
    else:
        scen_emis_list = {scen: scen for scen in scenario_list}
    
    logger.info(scen_emis_list)

    egrid, mapped_df, unmapped_df = None, None, None
    for scen_name, emis_name in scen_emis_list.items():
        logger.info(f"{scen_name}, {emis_name}")
        egrid, mapped_df, unmapped_df = processor.process_powerplant_scenario(
            scen_name, emis_name, nei_all_pt)

    processor.process_non_powerplant(
        scen_emis_list, unmapped_df, nei_all_pt)
    
    processor.create_non_powerplant_symlinks(scen_emis_list)

    # Plot 1: Base vs Final comparison
    processor.compare_emissions(scen_emis_list)

    # Plot 2: Compare shapefile emissions with original CSV
    processor.compare_with_original(scen_emis_list)

if __name__ == "__main__":
    import logging
    import yaml
    from datetime import datetime

    # start logger 
    logfile = f"log_files/run_electrification_emissions_{datetime.now():%Y%m%d_%H%M%S}.log"
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

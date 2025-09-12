import yaml
import geopandas as gpd
import os
import warnings
from LOCAETA_AQ.electrification_emissions_util import ElectrificationEmissionProcessor

# Suppress all warnings
warnings.filterwarnings('ignore')

def load_config(yaml_file: str) -> dict:
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():

    # Configuration
    cfg = load_config("config.yaml")
    config = cfg['electrification_emissions']

    overall_scenario = config["input"]["overall_scenario"]
    scenario_list = config["input"]["scenario_list"]

    # Create output directories if not exists
    os.makedirs(config['output']['output_dir'] + config["input"]["overall_scenario"], exist_ok=True)
    os.makedirs(config['output']['plots_dir']+ config["input"]["overall_scenario"], exist_ok=True)

    # Initialize processor
    processor = ElectrificationEmissionProcessor(config)

    # Read the point source emissions
    nei_all_pt = gpd.read_file(config['input']['combined_nei_file'])

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
    
    print(scen_emis_list)

    egrid, mapped_df, unmapped_df = None, None, None
    for scen_name, emis_name in scen_emis_list.items():
        print(scen_name, emis_name)
        egrid, mapped_df, unmapped_df = processor.process_powerplant_scenario(
            scen_name, emis_name, nei_all_pt, config)

    cs_emis = processor.process_non_powerplant(
        scen_emis_list, unmapped_df, nei_all_pt, config)

    # Plot 1: Base vs Final comparison
    processor.compare_emissions(scen_emis_list, config)

    # Plot 2: Compare shapefile emissions with original CSV
    processor.compare_with_original(scen_emis_list, config)

if __name__ == "__main__":
    main()

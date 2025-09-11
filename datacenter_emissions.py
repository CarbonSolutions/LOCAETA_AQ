import yaml
from LOCAETA_AQ.datacenter_emissions_util import DataCenterEmissionProcessor

def load_config(yaml_file: str) -> dict:
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():

    # Configuration
    cfg = load_config("config.yaml")
    config = cfg['datacenter_emissions']

    # Initialize processor
    processor = DataCenterEmissionProcessor(config)

    # create USA wide data center emissions
    processor.process_datacenter(config)

    # create regional data center emissions, if defined in config.yaml
    processor.process_subregional_emis(config)

    # plot total emissions for nei, base and final
    processor.run_datacenter_emissions_plots(config)

if __name__ == "__main__":
    main()

import yaml
import os
from LOCAETA_AQ.datacenter_emissions_utils import DataCenterEmissionProcessor
from LOCAETA_AQ.config_utils import load_config


def main(cfg):

    # Extract the relevant section from Configuration
    config = cfg['datacenter_emissions']
    # Add combined NEI file from cfg
    config['combined_nei_file'] = os.path.join(cfg['base_dirs']['nei_output_root'],
                                                cfg['nei_emissions']['output']['combined_pt_source_file'])

    combined_nei_file = cfg['nei_emissions']['output']['combined_pt_source_file']

    # Initialize processor
    processor = DataCenterEmissionProcessor(config)

    # create USA wide data center emissions
    processor.process_datacenter(config)

    # create regional data center emissions, if defined in config.yaml
    processor.process_subregional_emis(config)

    # plot total emissions for nei, base and final
    processor.run_datacenter_emissions_plots(config)

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    main(cfg)

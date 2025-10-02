import yaml
import os
from LOCAETA_AQ.datacenter_emissions_utils import DataCenterEmissionProcessor
from LOCAETA_AQ.config_utils import load_config
import logging

# logging from run_workflow 
logger = logging.getLogger(__name__)

def main(cfg):

    # Extract the relevant section from Configuration
    config = cfg['datacenter_emissions']
    # Add combined NEI file from cfg
    config['combined_nei_file'] = os.path.join(cfg['base_dirs']['nei_output_root'],
                                                cfg['nei_emissions']['output']['combined_pt_source_file'])

    # Initialize processor
    processor = DataCenterEmissionProcessor(config)

    # create USA wide data center emissions
    processor.process_datacenter()

    # create regional data center emissions, if defined in config.yaml
    processor.process_subregional_emis()

    # create symbolic link for "rest_NEI" emission file, whichs stays same among runs
    processor.create_rest_nei_emissions_symlinks()

    # plot total emissions for nei, base and final
    processor.run_datacenter_emissions_plots()

if __name__ == "__main__":

    import logging
    import yaml
    from datetime import datetime
    import argparse

    # start logger 
    logfile = f"log_files/run_datacenter_emissions_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,  # or DEBUG
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
           # logging.StreamHandler(),
            logging.FileHandler(logfile, mode="w")
        ]
    )
    
    parser = argparse.ArgumentParser(description="Run the datacenter emissions.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg)

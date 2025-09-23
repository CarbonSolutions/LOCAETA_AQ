import os
from LOCAETA_AQ.run_inmap_utils import INMAP_Processor
from LOCAETA_AQ.config_utils import load_config
import logging

# logging from run_workflow 
logger = logging.getLogger(__name__)


def main(cfg):

    # Initialize processor
    processor = INMAP_Processor(cfg)

    config = cfg["inmap"]
    scenario = cfg['stages']['scenario']
    run_names = cfg['stages']['run_names']

    logger.info(f"Creating inmap toml files for {scenario}: {run_names}")
    
    # --- Collect all runs ---
    run_infos = processor.collect_run_infos(scenario, run_names)
    
    logger.info(f"Checking run_infos for {run_infos}")

    # --- Generate TOML files ---
    for run_info in run_infos:
        processor.inmap_process_template(
            template_path=os.path.join(config['input']["toml_dir"], config['input']["toml_template"]),
            run_name=run_info["run_name"],
            nei_file_path=cfg['base_dirs']['nei_output_root'],
            emis_file_path=run_info["emis_file_path"],
            out_toml_path=run_info["inmap_run_file_output"]
        )

    # --- Build and save Bash script ---
    bash_script_path = os.path.join(cfg['base_dirs']['inmap_root'], f"run_inmap_{scenario}.sh")
    processor.generate_and_run_bash_script(cfg['base_dirs']['inmap_root'], run_infos, bash_script_path)


if __name__ == "__main__":

    import logging
    import yaml
    from datetime import datetime

    # start logger 
    logfile = f"run_inmap_{datetime.now():%Y%m%d_%H%M%S}.log"
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

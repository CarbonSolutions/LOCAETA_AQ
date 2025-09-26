import os
import warnings
from LOCAETA_AQ.run_benmap_utils import Benmap_Processor 
from LOCAETA_AQ.config_utils import load_config
import logging
import subprocess

# logging from run_workflow 
logger = logging.getLogger(__name__)

# Suppress all warnings
warnings.filterwarnings('ignore')


def main(cfg):

    # Initialize processor
    processor = Benmap_Processor(cfg)

    config = cfg["benmap"]
    scenario = cfg['stages']['scenario']
    run_names = cfg['stages']['run_names']
    
    # Benmap configuration
    benmap_root = config['input']['batchmode_dir']
    grid_level =  config['default_setup']['grid_level'] # 'county' # or 'tracts' 
    target_year = cfg["benmap"]['default_setup']['target_year'] # '2020'
    benmap_AQ_input_dir = cfg["benmap"]["input"]["input_dir"]
    ctlx_template = cfg["benmap"]["input"]["ctlx_template"]
    base_ctlx_template = cfg["benmap"]["input"]["base_ctlx_template"]

    # Wine executable path
    benmap_exe = cfg["benmap"]["input"]["benmap_exe"]
    default_base_run = cfg['inmap']['analyze']['default_base_run']

    # Collect run info
    output_pairs = processor.build_output_pairs(scenario, run_names, grid_level, target_year)

    # grid_level should be capitalize for BenMAP
    grid_level =  config['default_setup']['grid_level'].capitalize() # 'county' # or 'tracts' 

    # Key directories
    AQG_dir = benmap_root + '/AQG'
    CFG_dir = benmap_root +'/CFG'
    APV_dir = benmap_root +'/APV'

    # wine style directories
    CFGR_wine_dir = 'Z:'+ benmap_root + '/CFGR'
    APVR_wine_dir = 'Z:'+ benmap_root + '/APVR'

    process_once = False

    for run_name, paths in output_pairs.items():

        control_run = paths["sens"]
        base_run = paths["base"]

        logger.info(f"starting benmap model : {run_name}, base: {base_run}, control: {control_run}")

        # Skip re-processing the default base run after the first time
        if default_base_run in base_run and process_once:
            continue

        AQG_wine_dir = processor.convert_to_wine_path(AQG_dir)
        aqg_template = os.path.join(AQG_dir, base_ctlx_template)
        aqg_output = os.path.join(AQG_dir, f'{base_run}_wine.ctlx')
        processor.base_AQG_process_template(aqg_template, aqg_output, base_run, AQG_wine_dir, benmap_AQ_input_dir, grid_level)
        subprocess.run(["wine", benmap_exe, processor.convert_to_wine_path(aqg_output)], check=True)
        
        if default_base_run in base_run:
            process_once = True

        aqg_template = os.path.join(AQG_dir, ctlx_template)
        aqg_output = os.path.join(AQG_dir, f'{control_run}_wine.ctlx')
        processor.AQG_process_template(aqg_template, aqg_output, control_run, AQG_wine_dir, benmap_AQ_input_dir, grid_level)
        subprocess.run(["wine", benmap_exe, processor.convert_to_wine_path(aqg_output)], check=True)

        # Process and run CFG template
        CFG_wine_dir = processor.convert_to_wine_path(CFG_dir)
        cfg_template = os.path.join(CFG_dir, ctlx_template)
        cfg_output = os.path.join(CFG_dir, f'{control_run}_wine.ctlx')
        processor.CFG_process_template(cfg_template, cfg_output, control_run, base_run, CFG_wine_dir, CFGR_wine_dir, AQG_wine_dir)
        subprocess.run(["wine", benmap_exe, processor.convert_to_wine_path(cfg_output)], check=True)

        # Process and run APV template
        APV_wine_dir = processor.convert_to_wine_path(APV_dir)
        apv_template = os.path.join(APV_dir, ctlx_template)
        apv_output = os.path.join(APV_dir, f'{control_run}_wine.ctlx')
        processor.APV_process_template(apv_template, apv_output, control_run, APV_wine_dir, APVR_wine_dir, CFGR_wine_dir, grid_level)
        subprocess.run(["wine", benmap_exe, processor.convert_to_wine_path(apv_output)], check=True)

if __name__ == "__main__":
    import logging
    from datetime import datetime

    # start logger 
    logfile = f"log_files/run_benmap_{datetime.now():%Y%m%d_%H%M%S}.log"
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

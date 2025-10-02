import os
import subprocess
import sys
import logging
import json
import warnings
from LOCAETA_AQ.run_benmap_utils import Benmap_Processor 
from LOCAETA_AQ.config_utils import load_config
import tempfile
from quarto import render


# logging from run_workflow 
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def main(cfg):

    # Configuration 
    config = cfg["report"]
    main_root = cfg['base_dirs']['main_root']
    qmd_template = config["input"]["qmd_template"]
    report_run = cfg['stages']['report_run']
    output_dir = config["output"]["output_dir"]

    os.makedirs(output_dir, exist_ok=True)
    
    qmd_path = os.path.join(main_root, qmd_template)

    if not os.path.exists(qmd_path):
        print(f"❌ QMD template not found: {qmd_path}")
        sys.exit(1)

    if cfg['stages']['render_quarto_report']:
        try:
            report_output_file = f"quarto_report_for_{report_run}.html"


            # Build the command with -P flags for each parameter
            cmd = [
                "quarto", "render", qmd_path,
                "--output-dir", output_dir, 
                "--output", report_output_file 
            ]
            
            print(f"🔧 Running command: {' '.join(cmd)}")
            
            subprocess.run(cmd, check=True)
            print(f"✅ Report successfully rendered to: {report_output_file}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Quarto render failed with error code {e.returncode}")
            sys.exit(e.returncode)

if __name__ == "__main__":

    import logging
    import yaml
    from datetime import datetime
    import argparse

    # start logger 
    logfile = f"log_files/render_report_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,  # or DEBUG
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
           # logging.StreamHandler(),
            logging.FileHandler(logfile, mode="w")
        ]
    )
    
    parser = argparse.ArgumentParser(description="Rendering report.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg)
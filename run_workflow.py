import yaml
import subprocess
import logging
from pathlib import Path
import nei_emissions
import ccs_emissions
import datacenter_emissions
import electrification_emissions
import run_inmap, run_benmap
import analyze_benmap, analyze_inmap
import generate_report_results, render_report
from LOCAETA_AQ.config_utils import load_config
from datetime import datetime


# start logger 
logfile = f"log_files/run_workflow_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,  # or DEBUG
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.StreamHandler(),
        logging.FileHandler(logfile, mode="w")
    ]
)

def run_python_module(module, cfg):
    """Run the module's main function with cfg"""
    module.main(cfg)

def run_python(script, args=None):
    """Run a Python script with optional args"""
    cmd = ["python", script]
    if args:
        cmd.extend(args)
    logging.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main(cfg):

    scenario = cfg["stages"]["scenario"]

    # --- Step 1: Emissions ---
    logging.info(f"=== Step 1: Emission processing ({scenario}) ===")

    if cfg["stages"]["process_nei"]:
        run_python_module(nei_emissions, cfg) 

    if not cfg["stages"]["skip_process_scenario_emissions"]: 
        if scenario == "ccs_emissions":
            run_python_module(ccs_emissions, cfg) 

        elif scenario == "datacenter":
            run_python_module(datacenter_emissions, cfg)

        elif scenario == "electrification_emissions":
            run_python_module(electrification_emissions, cfg)

        elif scenario == "nei_emissions":
            logging.info("nei emissions should be ready")

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    # --- Step 2: Run INMAP ---
    if cfg["stages"]["run_inmap"]:
        logging.info(f"=== Step 2: Running INMAP ===")
        run_python_module(run_inmap, cfg)

    # --- Step 3: Analyze INMAP outputs and Convert to BenMAP + Run BenMAP---
    if cfg["stages"]["analyze_inmap"]:
        logging.info("=== Step 3: Analyzing INMAP outputs ===")
        run_python_module(analyze_inmap, cfg)

    # --- Step 4: Run BenMAP ---
    if cfg["stages"]["run_benmap"]:
        logging.info("=== Step 4: BenMAP processing ===")
        run_python_module(run_benmap, cfg)

    # --- Step 5: Analyze BenMAP outputs ---
    if cfg["stages"]["analyze_benmap"]:
        logging.info("=== Step 5: Analyzing BenMAP outputs ===")
        run_python_module(analyze_benmap, cfg)

    # --- Step 6: Generate combined results for reports ---
    if cfg["stages"]["generate_report_results"]:
        logging.info("=== Step 6: Generating combined run outputs ===")
        run_python_module(generate_report_results, cfg)

    # --- Step 7: Rendering quarto report ---
    if cfg["stages"]["render_quarto_report"]:
        logging.info("=== Step 7: Rendering quarto report ===")
        run_python_module(render_report, cfg)

    logging.info("=== Workflow completed successfully! ===")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Run the LOCAETA-AQ workflow.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg)
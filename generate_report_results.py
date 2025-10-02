
import os
import pandas as pd
import warnings
import logging
from LOCAETA_AQ.report_utils import report_processor
from LOCAETA_AQ.run_benmap_utils import Benmap_Processor 
from LOCAETA_AQ.config_utils import load_config

# logging from run_workflow 
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def main(cfg):

    # Initialize processor
    processor = report_processor(cfg)

    config = cfg["report"]
    scenario = cfg['stages']['scenario']
    report_run = cfg['stages']['report_run']
    run_names = cfg['stages']['run_names']
    grid_level =  cfg['benmap']['default_setup']['grid_level'] 
    target_year =cfg['benmap']['default_setup']['target_year'] 

    logger.info(f"Generating combine results for scenario : {scenario}")

    run_benmap_processor = Benmap_Processor(cfg)
    # Collect run info
    output_pairs = run_benmap_processor.build_output_pairs(scenario, run_names, grid_level, target_year)
    logger.info(f"starting {output_pairs}")
     
    # output_directories
    inmap_ouput_root = cfg["inmap"]["output"]["plots_dir"]
    benmap_output_root = cfg["benmap"]["output"]["plots_dir"]
    report_plots_dir = os.path.join(cfg["report"]["output"]["plots_dir"], report_run)
    os.makedirs(report_plots_dir, exist_ok=True) 

    combined_df = None

    inmap_target_file = 'area_weighted_averages.csv'
    inmap_final_output = 'area_weighted_averages_all_runs'

    benmap_output_type =['incidence' , 'valuation']

    run_names = output_pairs.keys()

    # Step 1: INMAP combined results
    for run in run_names:
        output_path = os.path.join(inmap_ouput_root, run)
        df = pd.read_csv(os.path.join(output_path, inmap_target_file))

        df.rename(columns={"Area-Weighted Average": run}, inplace=True)

        # Merge on 'Species' column
        if combined_df is None:
            combined_df = df  # First dataframe, set as base
        else:
            combined_df = pd.merge(combined_df, df, on="Species", how="outer")

    # Generate combined bar plots
    processor.plot_area_weighted_average_all_runs(combined_df, run_names, report_plots_dir, f"{inmap_final_output}.png") 
    # Save the mean concentrations from all runs in inmap_final_output
    processor.save_area_weighted_avg_for_all_runs (combined_df, report_plots_dir, f"{inmap_final_output}.csv")

    # Step 2: BenMAP combined results
    for benmap_output in benmap_output_type:
        combined_df = None
        combined_df_normalized = None

        benmap_target_file = f'{benmap_output}_Summary_Table_Health_Benefits_by_Race_in_Nation.csv'

        for run in run_names:
            logger.info(f"Processing run: {run}")
            df_mean, df_normalized = processor.read_and_prepare_benmap_csv(run, benmap_output_root, benmap_target_file, benmap_output)

            combined_df = processor.merge_benmap_dfs(combined_df, df_mean)
            if benmap_output == 'incidence' and df_normalized is not None:
                combined_df_normalized = processor.merge_benmap_dfs(combined_df_normalized, df_normalized)

        logger.info(f"Combined DataFrame for {benmap_output}:\n{combined_df}")
        if benmap_output == 'incidence':
            logger.info(f"Combined Normalized DataFrame for {benmap_output}:\n{combined_df_normalized}")

        # Save and plot results
        processor.save_and_plot_results(combined_df, combined_df_normalized, benmap_output, report_plots_dir)

    # Step 3: Linking overall emissions figure
    emis_plot_dir =os.path.join(cfg['base_dirs']['output_root'], 'emissions')

    if report_run in ['current_easyhard','current_easyhard_Food_Agr']: # "electrification_emissions"
        src_emis_plot_dir = os.path.join(emis_plot_dir, report_run)
        src_file_name = 'Total_Difference.png'

    elif report_run in ['current_2020', '2050_decarb95', '2050_noIRA_111D']:  # "datacenter_emissions" 
        src_emis_plot_dir = os.path.join(emis_plot_dir, report_run)
        src_file_name = 'Total_Difference.png'

    elif report_run in ['USA_CCS', 'LA_CCS', 'CO_CCS']: # == "ccs_emissions":
        src_emis_plot_dir = os.path.join(emis_plot_dir, report_run)
        src_file_name = "NEI_vs_CCS_vs_zero_out_emissions_all_industrial_facilities.png"
    else:
        raise ValueError (f"Error in generate_report_results: no matching emissions for {report_run}")
    
    src_plot_file = f"{src_emis_plot_dir}/{src_file_name}"
    if not os.path.exists(src_plot_file):
        raise FileNotFoundError(f"Source figure file not found: {src_plot_file}")

    dst_file = f"{report_plots_dir}/overall_emis_plot.png"
    # Check if the symlink exists before attempting to remove it
    if os.path.islink(dst_file):
        os.remove(dst_file)
        logger.info(f"Removed existing symlink '{dst_file}'")
    os.symlink(os.path.abspath(src_plot_file), dst_file)

if __name__ == "__main__":

    import logging
    import yaml
    from datetime import datetime
    import argparse

    # start logger 
    logfile = f"log_files/generate_report_results_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,  # or DEBUG
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
           # logging.StreamHandler(),
            logging.FileHandler(logfile, mode="w")
        ]
    )
    
    parser = argparse.ArgumentParser(description="Generate report results.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg)

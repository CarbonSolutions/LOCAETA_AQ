import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import logging
import warnings
import os
from .run_inmap_utils import INMAP_Processor

# logging from run_workflow 
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class Benmap_Processor:
    """
    A class to run BenMAP model
    """

    def __init__(self, cfg):
        """
        Initialize the processor with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with file paths and parameters
        """
        self.cfg = cfg


##############################################################################
####### The functions below are for running BenMAP batchmode           #######
##############################################################################

    def base_AQG_process_template(self, template_path, output_path, base_run, AQG_wine_dir, inmap_wine_dir, grid_level):
        """Reads a template file, replaces placeholders with actual paths, and saves the modified file."""
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Replace placeholders
        content = content.replace('{AQG_wine_dir}', AQG_wine_dir)
        content = content.replace('{inmap_wine_dir}', inmap_wine_dir)
        content = content.replace('{base_run}', base_run)
        content = content.replace('{grid_level}', grid_level)

        logger.info(f"Creating base AQG output: {output_path}")
        # Write the modified content to a new file
        with open(output_path, 'w') as f:
            f.write(content)

    def AQG_process_template(self, template_path, output_path, control_run, AQG_wine_dir, inmap_wine_dir, grid_level):
        """Reads a template file, replaces placeholders with actual paths, and saves the modified file."""
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Replace placeholders
        content = content.replace('{AQG_wine_dir}', AQG_wine_dir)
        content = content.replace('{inmap_wine_dir}', inmap_wine_dir)
        content = content.replace('{control_run}', control_run)
        content = content.replace('{grid_level}', grid_level)

        logger.info(f"Creating control AQG output: {output_path}")
        # Write the modified content to a new file
        with open(output_path, 'w') as f:
            f.write(content)


    def CFG_process_template(self, template_path, output_path, control_run, base_run, CFG_wine_dir, CFGR_wine_dir, AQG_wine_dir):
        """Reads a template file, replaces placeholders with actual paths, and saves the modified file."""
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Replace placeholders
        content = content.replace('{CFG_wine_dir}', CFG_wine_dir)
        content = content.replace('{CFGR_wine_dir}', CFGR_wine_dir)
        content = content.replace('{AQG_wine_dir}', AQG_wine_dir)
        content = content.replace('{control_run}', control_run)
        content = content.replace('{base_run}', base_run)

        logger.info(f"Creating CFG output: {output_path}")
        # Write the modified content to a new file
        with open(output_path, 'w') as f:
            f.write(content)

    def APV_process_template(self, template_path, output_path, control_run, APV_wine_dir, APVR_wine_dir, CFGR_wine_dir, grid_level):
        """Reads a template file, replaces placeholders with actual paths, and saves the modified file."""
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Replace placeholders
        content = content.replace('{APV_wine_dir}', APV_wine_dir)
        content = content.replace('{APVR_wine_dir}', APVR_wine_dir)
        content = content.replace('{CFGR_wine_dir}', CFGR_wine_dir)
        content = content.replace('{control_run}', control_run)
        content = content.replace('{grid_level}', grid_level)

        logger.info(f"Creating APV output: {output_path}")
        # Write the modified content to a new file
        with open(output_path, 'w') as f:
            f.write(content)

    # Function to convert Unix paths to Wine-compatible Windows paths
    def convert_to_wine_path(self, unix_path):
        windows_path = unix_path #s.replace('/', '\\')  # Convert to Windows-style backslashes
        return "Z:" + windows_path  # Prepend Z: for Wine

    def build_output_pairs(self, scenario, run_names, grid_level, target_year):
        """
        Build dictionary mapping each run_name to its base and sensitivity output CSV paths,
        without needing to call collect_analysis_infos() first.

        Parameters
        ----------
        scenario : str
            Scenario key (e.g., 'datacenter_emissions', 'ccs_emissions').
        run_names : list of str
            List of run names to process.
        grid_level : str
            Grid level string (e.g., 'county', 'tract').
        target_year : str or int
            Target year for naming outputs.

        Returns
        -------
        dict
            Dictionary of run_name -> {"base": <base_csv_filename>, "sens": <sens_csv_filename>}
        """
        output_pairs = {}
        run_inmap_obj = INMAP_Processor(self.cfg)

        for target_run_name in run_names:
            _, run_name = run_inmap_obj.get_emission_paths(scenario, target_run_name)

            # --- Base emissions ---
            if self.cfg[scenario]['has_own_base_emission']:
                base_run_name = f"{run_name}_base"
                base_output_csv_path = f"{base_run_name}_{grid_level}_inmap_{target_year}_pm25"
            else:
                default_base_run = self.cfg['inmap']['analyze']['default_base_run']
                base_output_csv_path = f"{default_base_run}_{grid_level}_inmap_{target_year}_pm25"

            # --- Sensitivity run ---
            sens_output_csv_path = f"control_{run_name}_{grid_level}_inmap_{target_year}_pm25"

            # --- Store main run ---
            output_pairs[run_name] = {
                "base": base_output_csv_path,
                "sens": sens_output_csv_path,
            }

            # --- Handle separate cases (if any) ---
            separate_cases = self.cfg.get('stages', {}).get('separate_case_per_each_run') or []
            for case_name in separate_cases:
                _, run_name_case = run_inmap_obj.get_emission_paths(scenario, f"{target_run_name}_{case_name}")
                sens_output_csv_path_case = f"control_{run_name_case}_{grid_level}_inmap_{target_year}_pm25"

                output_pairs[run_name_case] = {
                    "base": base_output_csv_path,
                    "sens": sens_output_csv_path_case,
                }

        # Exclude other runs if "run_only_separate_case" is true. 
        if self.cfg["stages"]["run_only_separate_case"]:
            separate_cases = self.cfg.get('stages', {}).get('separate_case_per_each_run') or []
            for run_name in list(output_pairs.keys()):
                if not any(case_name in run_name for case_name in separate_cases):
                    del output_pairs[run_name]

        return output_pairs
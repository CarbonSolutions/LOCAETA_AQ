import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import logging
import warnings
import os
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

        print("AQG debug", output_path)
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

        print("AQG debug", output_path)
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

        # Write the modified content to a new file
        with open(output_path, 'w') as f:
            f.write(content)

    # Function to convert Unix paths to Wine-compatible Windows paths
    def convert_to_wine_path(self, unix_path):
        windows_path = unix_path #s.replace('/', '\\')  # Convert to Windows-style backslashes
        return "Z:" + windows_path  # Prepend Z: for Wine

    def build_output_pairs(self, run_pairs, scenario, grid_level, target_year):
        """
        Build dictionary mapping each run_name to its base and sensitivity output CSV paths.

        Parameters
        ----------
        run_pairs : dict
            Dictionary of run_name -> {"base": <base shapefile path>, "sens": <sensitivity shapefile path>}
        cfg : dict
            Configuration dictionary.
        scenario : str
            Scenario name (used to check config).
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

        for run_name, paths in run_pairs.items():
            base_path = paths["base"]

            # --- Base run ---
            if not self.cfg[scenario]['has_own_base_emission']:
                default_base_run = self.cfg['inmap']['analyze']['default_base_run']
                if default_base_run not in base_path:
                    raise ValueError(f"The file is NOT an expected base run: {base_path}")
                base_output_csv_path = f"{default_base_run}_{grid_level}_inmap_{target_year}_pm25"
            else:
                base_run_name = os.path.basename(os.path.dirname(base_path))
                if "base" in base_run_name:
                    base_output_csv_path = f"{base_run_name}_{grid_level}_inmap_{target_year}_pm25"
                else:
                    raise FileExistsError(f"The file name is NOT an expected base run: {base_run_name}" )
            # --- Sensitivity run ---
            sens_output_csv_path = f"control_{run_name}_{grid_level}_inmap_{target_year}_pm25"

            # --- Store in dictionary ---
            output_pairs[run_name] = {
                "base": base_output_csv_path,
                "sens": sens_output_csv_path,
            }

        return output_pairs
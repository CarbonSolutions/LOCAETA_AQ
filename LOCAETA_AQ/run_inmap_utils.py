import os
import subprocess
import glob
import logging
# Suppress all warnings in jupyter notebook
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class INMAP_Processor:
    """
    A class to run INMAP model
    """

    def __init__(self, cfg):
        """
        Initialize the processor with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with file paths and parameters
        """
        self.cfg = cfg

    def get_emission_paths(self, get_emission, run_name):
        """
        Return output_base and validated run_names for a given emission type.

        Parameters
        ----------
        get_emission : str
            Emission type key (e.g., 'ccs_emissions', 'datacenter_emissions', 'electrification_emissions').
        run_name : str

        Returns
        -------
        tuple
            (output_base, valid_run_names)
        """
        config = self.cfg[get_emission]

        # Build the list of allowed run names for validation
        allowed = set()

        if get_emission == "ccs_emissions":
            # target_scenario is a single string
            base = config.get("target_scenario")
            if base:
                allowed.add(base)
                allowed.add(f"{base}_wo_NH3_VOC")

            for case in config.get("separate_scenario", []):
                ctype = case.get("type")
                if ctype == "industrial_no_ccs":
                    allowed.add(f"{base}_without_CCS_facilities")
                elif ctype in {"state_all_facilities", "state_specific_facilities"}:
                    name = case.get("name")
                    if name:
                        allowed.add(f"{name}_CCS")
                        allowed.add(f"{name}_CCS_wo_NH3_VOC")

        elif get_emission == "datacenter_emissions":
            for scen in config.get("target_scenario", []):
                scen = str(scen)
                allowed.add(scen)
                allowed.add(f"{scen}_base")
                for region in config.get("separate_scenario", []):
                    allowed.add(f"{scen}_{region}")

        elif get_emission == "electrification_emissions":
            
            for overall_scenario in ['Full_USA', 'Food_Agr']:
                for scen in config.get("target_scenario", []):
                    scen = str(scen)

                    if overall_scenario != "Full_USA":
                        scen = f"{str(scen)}_{overall_scenario}"

                    allowed.add(scen)
                    allowed.add(f"{scen}_base")

        elif get_emission == "nei_emissions":
            # target_scenario is a single string
            base = config.get("target_scenario")
            if base:
                allowed.add(base)

        # Validate
        if run_name not in allowed:
            raise ValueError(f"Invalid run_names {run_name} for {get_emission}. Allowed: {sorted(allowed)}")

        # Output base path
        output_base = config["output"]["output_dir"]
        if config.get("master_scenario"):
            output_base = os.path.join(output_base, config["master_scenario"])

        return output_base, run_name

    def inmap_process_template(self, template_path,run_name, nei_file_path, emis_file_path, out_toml_path):

        """
        Generate a TOML configuration file for INMAP by replacing placeholders in a template.

        Parameters
        ----------
        template_path : str
            Path to the input TOML template.
        run_name : str
            Name of the run to insert into the template.
        nei_file_path : str
            Path to the NEI input file to insert into the template.
        emis_file_path : str
            Long string with a list of emission shapefiles, formatted for INMAP, to insert into the template.
        out_toml_path : str
            Path where the generated TOML file will be saved.

        Returns
        -------
        None
        """

        with open(template_path, 'r') as f:
            content = f.read()
        
        # Replace placeholders
        content = content.replace('{nei_file_path}', nei_file_path)
        content = content.replace('{emis_file_path}', emis_file_path)
        content = content.replace('{run_name}', run_name)

        with open(out_toml_path, 'w') as f:
            f.write(content)

    def get_emis_file_list(self, output_base, run_name):
        """
        Find all shapefiles for a given run and and format the list into a long string

        Parameters
        ----------
        output_base : str
            Base directory where emission outputs are stored.
        run_name : str
            Name of the run to look for shapefiles.

        Returns
        -------
        str
            A string of absolute paths to shapefiles, each quoted and separated by ', \n'.
        """
        pattern = os.path.join(output_base, run_name, "**", "*.shp")

        matches = glob.glob(pattern, recursive=True)

        # Format as requested
        emis_file_path = ", \n".join(f"'{os.path.abspath(f)}'" for f in matches)
        return emis_file_path

    def generate_and_run_bash_script(self, inmap_root, run_infos, bash_script_path):

        """
        Generate a Bash script to run INMAP for multiple runs and execute it.

        Parameters
        ----------
        inmap_root : str
            Root directory where INMAP executable is located.
        run_infos : list of dict
            List of run information dictionaries. Each dict must contain keys:
            'run_name' and 'inmap_run_file_output'.
        bash_script_path : str
            Path where the Bash script will be saved.

        Returns
        -------
        None
        """
            
        # --- Prepare the runs list ---
        runs_str = "\n  ".join([f'"{r["run_name"]}"' for r in run_infos])
        toml_mapping_str = "\n  ".join([f'"{r["inmap_run_file_output"]}"' for r in run_infos])

        # Build the Bash script
        bash_template = f"""#!/bin/bash

cd "{inmap_root}" || exit 1

runs=(
{runs_str}
)

tomls=(
{toml_mapping_str}
)

for i in "${{!runs[@]}}"; do
    run_name="${{runs[$i]}}"
    toml_path="${{tomls[$i]}}"

    echo "=== Starting run: $run_name ==="
    mkdir -p "outputs/$run_name"
#    ./inmap run steady -s --config "$toml_path"
    echo "=== Finished run: $run_name ==="
done
"""

        with open(bash_script_path, "w") as f:
            f.write(bash_template)

        os.chmod(bash_script_path, 0o755)

        # --- Execute Bash script ---
        subprocess.run(["caffeinate", "-i", bash_script_path], check=True)

    def collect_run_infos(self, scenario, run_names):
        """
        Collect information for multiple INMAP runs, including emission shapefiles 
        and output TOML file paths.

        Parameters
        ----------
        scenario : str
            Scenario key (e.g., 'datacenter_emissions', 'ccs_emissions').
        run_names : list of str
            List of run names to process.

        Returns
        -------
        list of dict
            Each dict contains keys:
                - 'run_name': str
                - 'emis_file_path': str
                - 'inmap_run_file_output': str
        """
        run_infos = []



        for target_run_name in run_names:
            output_base, run_name = self.get_emission_paths(scenario, target_run_name)

            if not self.cfg["stages"]["run_only_separate_case"]: 
                emis_file_path = self.get_emis_file_list(output_base, run_name)
                inmap_run_file_output = os.path.join(self.cfg["inmap"]["input"]["toml_dir"], f"nei2020Config_{run_name}.toml")
                run_infos.append({"run_name": run_name, "emis_file_path": emis_file_path, "inmap_run_file_output": inmap_run_file_output})

            # Base emissions
            if self.cfg[scenario]['has_own_base_emission']:
                if not self.cfg["stages"]["skip_base_run"]: 
                    base_run_name = run_name + "_base"
                    emis_file_path_base = self.get_emis_file_list(output_base, base_run_name)
                    inmap_run_file_output_base = os.path.join(self.cfg["inmap"]["input"]["toml_dir"], f"nei2020Config_{base_run_name}.toml")
                    run_infos.append({"run_name": base_run_name, "emis_file_path": emis_file_path_base, "inmap_run_file_output": inmap_run_file_output_base})
    
            # Separate cases
            separate_cases = self.cfg.get('stages', {}).get('separate_case_per_each_run') or []
            for case_name in separate_cases:
            #for case_name in self.cfg['stages'].get('separate_case_per_each_run', []):
                output_base, run_name_case = self.get_emission_paths(scenario, f"{target_run_name}_{case_name}")
                emis_file_path_case = self.get_emis_file_list(output_base, run_name_case)
                inmap_run_file_output_case = os.path.join(self.cfg["inmap"]["input"]["toml_dir"], f"nei2020Config_{run_name_case}.toml")
                run_infos.append({"run_name": run_name_case, "emis_file_path": emis_file_path_case, "inmap_run_file_output": inmap_run_file_output_case})

        return run_infos

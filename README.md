# Local Climate and Air Emissions Tracking Atlas (LOCAETA)

LOCAETA is an interactive, user-friendly data platform that integrates advanced atmospheric models and datasets to quantify the air quality and public health impacts of decarbonization technologies.

## Background

The [LOCAETA Data Explorer](https://apps.carbonsolutionsllc.com/locaeta/) provides accessible, science-based air quality information to help communities, policymakers, and researchers understand how decarbonization strategies affect local air quality and health outcomes.  

It combines:

- Satellite and in situ air quality observations  
- Industrial facility-level emissions  
- Screening-level modeling of air quality and health impacts  

### Decarbonization Strategies Analyzed

- **Carbon Capture and Storage (CCS)**
- **Industrial Electrification**
- **Data Center Electrification**
- **Fuel Switching (e.g., Hydrogen, Clean Fuels)**

Each pathway includes facility-level emission changes, corresponding air quality modeling with INMAP, and public health impact assessments using BenMAP.

---

## Repository Overview

This repository, **LOCAETA_AQ**, contains the code and configurations for automated emission processing, air quality modeling, and health impact workflows.  

### Core Components

- **Emissions** – Processing raw NEI-SMOKE formatted csv files to shapefiles 
- **INMAP** – Reduced-form air quality model ([details](https://inmap.run/))
- **BenMAP** – Health impact model ([details](https://www.epa.gov/benmap))
- **Quarto Reports** – Automated report generation from model outputs

### Repository Structure

```plaintext
LOCAETA_AQ/
├── config.yaml                      # Central configuration file controlling the full workflow
├── run_workflow.py                  # Main orchestration script (runs all workflow stages sequentially)
│
├── nei_emissions.py                 # Process NEI base emissions
├── ccs_emissions.py                 # Process CCS emissions
├── datacenter_emissions.py          # Process data center emissions
├── electrification_emissions.py     # Process industrial electrification emissions
│
├── run_inmap.py                     # Run INMAP simulations
├── analyze_inmap.py                 # Analyze INMAP outputs (maps, summaries, comparisons)
│
├── run_benmap.py                    # Run BenMAP simulations (batch mode)
├── analyze_benmap.py                # Analyze BenMAP outputs and compute health metrics
│
├── generate_report_results.py       # Combine INMAP + BenMAP outputs for report input
├── render_report.py                 # Render automated Quarto reports (HTML/PDF)
│
├── quarto_reports/                  # Quarto templates and rendered reports
├── outputs/                         # Model outputs (excluded from repo)
├── LOCAETA_AQ/                      # Helper functions and shared utilities
├── setup.py
└── README.md
```

---

# Config-Driven Workflow

Starting from this version, the **entire LOCAETA-AQ workflow**—from emissions processing to report generation—is automated using the configuration file `config.yaml` and a single driver script `run_workflow.py`. Note that each workflow driver (e.g., run_inmap, analyze_benmap) can be executed individually. 

---

## 1. Overview

The workflow proceeds in stages:

1. **Emission Processing**
2. **INMAP Simulation**
3. **INMAP Analysis**
4. **BenMAP Health Impact Analysis** (This is written to run in Mac OS)
5. **BenMAP Analysis**
6. **Report Generation (Quarto)**

Each stage is controlled by the `stages` section in `config.yaml`.

---

## 2. Configure the Workflow (`config.yaml`)

All user-editable settings are in [`config.yaml`](./config.yaml).  
Key sections:

### 🔹 `base_dirs`
Defines the main directory structure:
```yaml
base_dirs:
  main_root: "/path/to/LOCAETA_AQ"
  output_root: "/path/to/LOCAETA_AQ/outputs"
  nei_root: "/path/to/NEI_emissions"
  ccs_root: "/path/to/CCS_emissions"
  inmap_root: "/path/to/inmap-1.9.6-gridsplit"
  benmap_root: "/path/to/BenMAP"
  report_root: "/path/to/quarto_reports"
```

### 🔹 `stages`
Controls what to run:
```yaml
stages:
  scenario: datacenter_emissions        # Select emission scenario type
  run_names: 
    - current_2020                      # Emission scenario to process
  separate_case_per_each_run:
    - MISO_Central                      # Regional subcase (optional)

  process_nei: false                    # Process NEI base emissions
  skip_process_scenario_emissions: true # Skip emission reprocessing if done
  skip_base_run: true                   # Skip INMAP base if already available
  run_only_separate_case: true          

  run_inmap: true
  analyze_inmap: true
  run_benmap: true
  analyze_benmap: true
  generate_report_results: true
  render_quarto_report: true
  report_run: current_2020
```

### 🔹 Scenario Options
Each emission scenario (e.g., `ccs_emissions`, `datacenter_emissions`, `electrification_emissions`) defines its own input/output settings, target scenarios, and notes.  
For example:

```yaml
datacenter_emissions:
  input:
    raw_csv_dir: "."
  output:
    output_dir: output
    plots_dir: emissions
  target_scenario:
    - current_2020
    - 2050_noIRA_111D
    - 2050_decarb95
  separate_scenario:
    - CAISO
    - PJM_East
    - MISO_Central
  has_own_base_emission: true
```

You can switch between emission types and run_names by setting:
```yaml
stages:
  scenario: ccs_emissions
  run_names: 
  - current_2020                     
  separate_case_per_each_run:
  - MISO_Central  
```
or
```yaml
stages:
  scenario: electrification_emissions
  run_names: 
  - current_easyhard
  - 2050_easyhard_decarb95                      
```

---

## 3. Run the Workflow

Once the configuration file is ready, simply execute:

```bash
python run_workflow.py --config config.yaml
```

This will:
1. Read all parameters from `config.yaml`
2. Automatically execute each modeling and analysis stage based on the flags under `stages`
3. Save intermediate and final outputs under the appropriate directories defined in `base_dirs` and `output_dir` and `plots_dir` under Scenario options

---

## 4. Example Runs

### Example 1: Datacenter Scenario (run only Cambium region runs, so it uses skip_base_run and run_only_separate_case)
```yaml
stages:
  scenario: datacenter_emissions
  run_names: 
    - current_2020
    - 2050_decarb95
  separate_case_per_each_run: 
    - CAISO
    - PJM_East
    - MISO_Central
    - MISO_South
    - NorthernGrid_West
    - SPP_North
  skip_base_run: false
  run_only_separate_case: true
  run_inmap: true
  analyze_inmap: true
  run_benmap: true
  analyze_benmap: true
```

### Example 2: CCS USA scenario only (skip base INMAP run, which is NEI2020 run)
```yaml
stages:
  scenario: ccs_emissions
  run_names: 
    - USA_CCS
    - USA_CCS_wo_NH3_VOC
    - USA_zero_out_CCS
  skip_base_run: true
  run_only_separate_case: false
  run_inmap: true
  analyze_inmap: true
  run_benmap: true
  analyze_benmap: true
```

### Example 3: Electrification scenario only
```yaml
stages:
  scenario: electrification_emissions
  run_names: 
    - current_easyhard_Food_Agr
    - 2050_easyhard_noIRA_111D_Food_Agr
    - 2050_easyhard_decarb95_Food_Agr
  skip_base_run: false
  run_only_separate_case: false
  run_inmap: true
  analyze_inmap: true
  run_benmap: true
  analyze_benmap: true

```

## Description of available runs from the current repository

<p align="center">
  <img src="./Run_descriptions.pdf" alt="LOCAETA-AQ Run Description Table" width="85%">
</p>

**Figure:** Description of all available runs from the current LOCAETA-AQ repository 


---

## 5. Outputs

After each run, results are organized under:
```
outputs/
├── emissions/
├── inmap/
├── benmap/
└── report/
```

### INMAP
- Concentration maps, CSVs, and JSON summaries saved under `output_root/inmap`
- Plots and spatial summaries in `plots_dir`

### BenMAP
- Health impact results and summaries in `benmap_root/APVR`
- Plots under `output_root/benmap`

### Quarto Reports
- Automatically generated HTML reports under `report_root/final_reports`
- Based on `LOCAETA_report_template.qmd`

---

## 6. Troubleshooting

- Make sure directory paths in `config.yaml` exist before running.
- If you only want to analyze existing results, set:
  ```yaml
  run_inmap: false
  run_benmap: false
  analyze_inmap: true
  analyze_benmap: true
  ```
- To re-run only report generation:
  ```yaml
  generate_report_results: true
  render_quarto_report: true
  ```

---

## Contact

**Yunha Lee**  
*Research Scientist, Carbon Solutions LLC*  
📧 [yunha.lee@carbonsolutionsllc.com](mailto:yunha.lee@carbonsolutionsllc.com)

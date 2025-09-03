# Local Climate and Air Emissions Tracking Atlas (LOCAETA)

LOCAETA is an interactive, user-friendly data platform that utilizes a suite of cutting-edge atmospheric datasets (ground-based and remote sensing) and models to demonstrate the impact of decarbonization technologies on local air quality and public health.

## Background

The [LOCAETA Data Explorer](https://apps.carbonsolutionsllc.com/locaeta/) is designed to offer accessible air quality information in a user-friendly format, aimed at helping the public better understand how local air quality and public health are influenced by innovative decarbonization technologies. This platform provides:

- Satellite and in situ sensor data
- Identification of industrial facilities impacting community air quality
- Screening-level air quality modeling across the U.S.

#### Decarbonization Strategies Available to Explore: 

- Carbon Capture and Storage (CCS)
- Industrial Electrification
- Fuel Switching to Hydrogen or Other Clean Fuels

Each decarbonization option includes calculations of corresponding air quality co-benefits at the facility level, along with estimates of public health benefits for communities across a wide geographic area.

## LOCAETA-AQ repository Contents

This repository contains the code used for LOCAETA's air quality modeling and impact analysis. Currently, it uses the INMAP air quality model and performs a detailed public health assessment with well-established methods such as BenMAP.

- **INMAP Air Quality Model**: See the details of INMAP in this [link](https://inmap.run/)
- **BenMAP Public Health Model**: See the details of BenMAP in this [link](https://www.epa.gov/benmap/)


## Repository structure

```plaintext
LOCAETA_AQ/
├── LOCAETA_AQ/
├── quarto-reports/
│   ├── *.qmd                     # Quarto report files
│   ├── *.html                    # final HTML report
├── notebooks/                    # Directory for useful jupyter notebook iles for additional processing
│   ├── BenMAP_output_analysis_final.ipynb                     # Python script for BenMAP output analysis
│   ├── check_emissions.ipynb                                  # Python script for evaluating emissions
│   ├── DataCenter_to_NEI_SMOKE_emissions.ipynb                # Python script for DataCenter emission processing
│   ├── Electrification_to_NEI_SMOKE_emissions.ipynb           # Python script for Electrification emission processing
│   ├── USA_CCS_emission_processing.ipynb                      # Python script for USA CCS emission processing
│   ├── INMAP_further_analysis.ipynb                           # Python script for INMAP further analysis (plotting combined bar plots among runs)
│   ├── remove_NH3_VOC_from_CCS.ipynb                          # Jupyter notebook for creating CCS without NH3 
│   ├── modify_facility_emissions.ipynb                        # Jupyter notebook for modifying a single facility emission
├── outputs/                      # Directory for storing generated outputs (NOT INCLUDED IN THE REPO)
├── __init__.py
├── setup.py                                   
├── README.md                     # README file for project documentation                 
├── NEI_csv_to_shapefile.py       # Python script for converting NEI CSV data to shapefiles 
├── Incorporate_CCS_to_NEI.py     # Python script to incorporate CCS data into NEI dataset 
├── inmap_run_comparison.py       # Python script for comparing INMAP model runs 
├── process_INMAP_for_BenMAP.py   # Python script for processing INMAP data for BenMAP
└── README.md                     # README file for project documentation
```

# Worflow of LOCAETA-AQ


## Step 1: Emission processing

1. Generate NEI shapefiles from SMOKE ready csv files
```
python ./NEI_csv_to_shapefile.py
```
Note that it needs to set a directory where the SMOKE ready csv files are located and a directory where shapefiles will be saved. 

2. Apply new emission scenario into NEI shapefiles

#### **Regional amine-based CCS scenarios** 
```
python ./Incorporate_CCS_to_NEI.py
```
This script generate a new emission shapefile that has all point-source NEI emissions and a emission shapefile that includes CCS emissions for the facilities affected by CCS tech. The combined point-source emissions are needed, because CCS tech can apply to any point source type. With that, this script doesn't change non-point source. This script and the script called by this script may need to be revised, if the CCS output from Kelly is changed (esp. for column names). 

If you want to check/validate emissions, you can use this jupyter notebook: 
```
./notebooks/check_emissions.ipynb.
```

If you need to remove NH3 and VOC emission, you can use this jupyter notebook: 
```
./notebooks/remove_NH3_VOC_from_CCS.ipynb
```

If you need to modify a specific facility emissions, you can use this jupyter notebook: 
```
./notebooks/modify_facility_emissions.ipynb
```

#### **Whole USA amine-based CCS scenarios**
```
./notebooks/USA_CCS_emission_processing.ipynb
```
Note that the regional CCS emission processing is developed first but the whole USA CCS scenario is much complicated, so I wrote new script to process the whole USA CCS emissions. 

#### **Data Center scenarios** 
```
./notebooks/DataCenter_to_NEI_SMOKE_emissions.ipynb
```

#### **Electrification scenarios** 
```
./notebooks/Electrification_to_NEI_SMOKE_emissions.ipynb
```

## Step 2:  Run INMAP

Once the emission files are processed, it is time to run INMAP. You can find the instruction on how to run INMAP in this repo: https://github.com/yunhal/inmap-1.9.6-gridsplit 

Brief steps: 
  1. create "toml" run file (you can modify eval/nei2020Config_CO_CCS.toml)
  2. set the emission file path you just processed
  3. set the INMAP output file path (make sure to create the output directory as well)
  4. run INMAP  

You can use this script (inmap-1.9.6-gridsplit/create_inmap_toml.ipynb) to run these steps in one shot.

## Step 3: Analyze INMAP runs
```
python ./inmap_run_comparison.py 
```
Make sure to set run_pairs which defines the INMAP output file path and scenario name. This script will generate figures under analysis_output_dir and json files under webdata_path by each run pair.

To general a combined bar plot with multiple INMAP runs, do this: 

```
./notebooks/INMAP_further_analysis.ipynb 
```

## Step 4: Convert INMAP output to BenMAP input file and run BenMAP
```
python ./process_INMAP_for_BenMAP.py
```
This will generate air quality files in csv for each INMAP run, which will be used as an input to BenMAP. The instruction to run BenMAP will be available in this repo: https://github.com/yunhal/BenMAP_batchmode_for_MAC

In that repo, you will find this script (batchmode/run_benmap_Wine.ipynb) that will run BenMAP in batch mode. 

## Step 5: Analyze BenMAP ouputs

You can run this jupyter notebook for that: 
```
./notebooks/BenMAP_output_analysis_final.ipynb
```

## Step 6: Generate a Quarto report

I designed a quarto file to generate a LOCAETA-AQ report with minimum efforts. It will automatically grab key INMAP/BenMAP results and scenario descriptions files. For instance, here are the key model results used in the report: 1) a total emission plot that describes the overall emission changes in the scenario, 2) overall area-weighted concentrations from INMAP, 3) spatial distribution of total PM2.5 from INMAP, 4) BenMAP outputs. 

The scenario description files (i.e., intro.txt, simulations_description.csv, and emis_description.txt) must be available under "outputs/report_txt/{run_name}" before generating a report. 


## Contact Information

For any questions or further information regarding this repository, please contact:

**Yunha Lee**  
**Research Scientist**  
**Carbon Solutions**  
Email: [yunha.lee@carbonsolutionsllc.com](mailto:yunha.lee@carbonsolutionsllc.com)

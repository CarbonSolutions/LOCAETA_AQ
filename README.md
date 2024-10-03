# Local Climate and Air Emissions Tracking Atlas (LOCAETA)

LOCAETA is an interactive, user-friendly data platform that utilizes a suite of cutting-edge atmospheric datasets (ground-based and remote sensing) and models to demonstrate the impact of decarbonization technologies on local air quality and public health.

## LOCAETA Data Explorer

The [LOCAETA Data Explorer](https://apps.carbonsolutionsllc.com/locaeta/) is designed to offer accessible air quality information in a user-friendly format, aimed at helping the public better understand how local air quality and public health are influenced by innovative decarbonization technologies. This platform provides:

- Satellite and in situ sensor data
- Identification of industrial facilities impacting community air quality
- Screening-level air quality modeling across the U.S.

### Decarbonization Strategies Available to Explore: 

- Carbon Capture and Storage (CCS)
- Industrial Electrification
- Fuel Switching to Hydrogen or Other Clean Fuels

Each decarbonization option includes calculations of corresponding air quality co-benefits at the facility level, along with estimates of public health benefits for communities across a wide geographic area.

## Repository Contents

This repository contains the code used for LOCAETA's air quality modeling and impact analysis. Currently, it uses the INMAP air quality model and performs a detailed public health assessment with well-established methods such as BenMAP.

- **INMAP Air Quality Model**: See the details of INMAP [here](https://inmap.run/)
- **Public Health Assessment**: Utilizes methods like BenMAP for comprehensive analysis.


## Repository structure

LOCAETA_AQ/
├── LOCAETA_AQ/
├── LOCAETA-reports/
│   ├── _build/                  # Directory containing built outputs for the Jupyter Book
│   ├── _config.yml              # Configuration file for Jupyter Book
│   ├── _toc.yml                 # Table of contents for Jupyter Book structure
│   ├── benmap_analysis.ipynb     # Jupyter notebook for BenMAP analysis
│   ├── emission_analysis.ipynb   # Jupyter notebook for CCS emissions analysis
│   ├── inmap_analysis.ipynb      # Jupyter notebook for INMAP analysis
│   ├── intro.md                 # Introduction markdown file for the LOCAETA project
│   ├── logo.png                 # Logo image for the project (NOT USED)
│   ├── references.bib           # Bibliography file for references (BibTeX format; NOT USED)
│   ├── requirements.txt         # Dependencies or packages required for the project
├── no_need/                     # Directory for files that are no longer needed
├── outputs/                     # Directory for storing generated outputs (NOT INCLUDED IN THE REPO)
├── __init__.py
├── setup.py                                   
├── README.md                     # README file for project documentation                 
├── NEI_csv_to_shapefile.py       # Python script for converting NEI CSV data to shapefiles (Step 1: Prep Emission files )
├── Incorporate_CCS_to_NEI.py     # Python script to incorporate CCS data into NEI dataset (Step 2: Apply CCS tech to emissions)
├── Amine_based_CCS_withoutNH3.ipynb  # Jupyter notebook to create amine-based CCS without NH3 (Step 2-2: Modify CCS without NH3 increase)
├── inmap_run_comparison.py       # Python script for comparing INMAP model runs (Step 3: compare a set of INMAP outputs)
├── process_INMAP_for_BenMAP.py   # Python script for processing INMAP data for BenMAP (Step 4: Create BenMAP input using INMAP outputs)
└── README.md                     # README file for project documentation

NOTE : After Step 2 (emissions are processed), INMAP should be run with the processed emissions files. See the details in this repo: https://github.com/yunhal/inmap-1.9.6-gridsplit

## Contact Information

For any questions or further information regarding this repository, please contact:

**Yunha Lee**  
**Research Scientist**  
**yunha.lee@carbonsolutionsllc.com**  
**Carbon Solutions**


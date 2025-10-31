# bonney_et-al_2025_msd-live

Kirk Bonney<sup>1\*</sup>, Nicole D. Jackson<sup>1</sup>, Stephen Ferencz<sup>2</sup>, Thushara Gunda<sup>1</sup>, and Raquel Valdez<sup>1\*</sup>

<sup>1 </sup> Sandia National Laboratories, Albuquerque, NM, USA
<sup>2 </sup> Pacific Northwest National Laboratory, Richland, WA, USA

\* corresponding author: klbonne@sandia.gov

## Overview
This metarepo contains the utilities and scripts used to generate a dataset of synthetic streamflows and corresponding water management outputs from the Water Rights Analysis Package (WRAP). The purpose of this repository is to provide the means to reproduce this dataset as well as to document the process by which the dataset was generated.

## Code reference
TODO

## Data reference
| Dataset                                                                          | Link                                                                                          | DOI              |
|----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|------------------|
| Synthetic Streamflow Datasets to Support Emulation of Water Allocations via LSTM | https://data.msdlive.org/records/cfj8d-xsb13                                                  | 10.57931/2441443 |
| Water Availability Model for the Colorado River Basin                            | https://www.tceq.texas.gov/permitting/water_rights/wr_technical-resources/wam.html            | n/a              |
| Water Rights for the Colorado River Basin                                        | https://tceq.maps.arcgis.com/apps/webappviewer/index.html?id=44adc80d90b749cb85cf39e04027dbdc | n/a              |

## Reproduce this work
Clone this repository (`git clone https://github.com/IMMM-SFA/bonney_et-al_2025_msd-live.git`) and install the `toolkit` package into a Python 3.11 environment (`pip install -e .`). Copy the data/ folder from the accompanying [MSD-Live archive](TODO) as a top level directory in the repository. Once the environment is established, this work can be reproduced by running scripts from the workflow/ directory. There are four subdirectories which correspond to different stages of the experiment:

| Directory name              | Description                                                                                                                                                                                                                                   |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| I_9505_Data_Preparation/    | This directory contains scripts for downloading and processing the 9505 dataset, associating streamflow reaches with control points, and preparing the data for HMM training.                                                                  |
| II_HMM_Streamflow_Generation/ | This directory contains scripts for training Bayesian Hidden Markov Models on historical streamflow data, generating synthetic streamflow ensembles, and exploring the generated data.                                                       |
| III_WRAP_Execution/         | This directory contains scripts for executing WRAP simulations using synthetic streamflow data and processing water management outputs.                                                                                                      |
| IV_Finalize_Dataset/        | This directory contains scripts for validating the final datasets and optimizing NetCDF files for distribution.                                                                                                                              |

Generally, all scripts have two header sections that the user will customize based on which datasets they wish to produce/use. The first is indicated by "## Settings ##" and allows the user to customize variables such as random seeds and parameters of the experiment. The second section is indicated by "## Path configuration ##" and includes paths to various input and output folders used in the script. Additional configuration is often imported from .json files in the `data/configs/` folder; these files can also be modified to change the experiment behavior. The intent is that the user should not need to customize these paths as long as they copy the data/ directory into the repository. Note that a large portion of the code is in functions and classes defined in the `toolkit` package included in the repository and users looking for implementation details should explore this package in addition to the scripts.

### Reproduce datasets
The workflow in `I_9505_Data_Preparation/` downloads and processes the 9505 dataset, which contains historical streamflow data from multiple climate models and scenarios. This data is then associated with specific river reaches and control points for use in subsequent HMM training and WRAP simulation workflows.

| Script name                         | Description                                                                                                                                                                                                                                                                                                            |
|-------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| i_download_data.py                  | This script downloads the raw 9505 data from the HydroSource2 website, filtering for specific HUC2 codes (11, 12, 13) corresponding to the basins of interest. The script creates organized folder structures for the downloaded NetCDF files. |
| ii_associate_pcp_and_reaches.py     | This script associates primary control points (PCPs) used by the WRAP simulator with the nearest river reaches using spatial analysis. It loads reach shapefiles and gage location shapefiles, then finds the nearest reach to each control point, saving the results to shapefiles and CSV files. |
| iii_subset_data_to_reaches.py      | This script subsets the 9505 data to only the reaches of interest, processing each HUC8 file to extract streamflow data for specific COMIDs (river reach identifiers) and saving the results to NetCDF files. |
| iv_combine_nc_files_and_convert_units.py | This script combines the subsetted 9505 data into a single NetCDF file and converts streamflow units from cubic feet per second (CFS) to acre-feet. It extracts metadata from filenames and organizes the data by time periods and scenarios. |

### Reproduce HMM training and synthetic streamflow generation
The workflow in `II_HMM_Streamflow_Generation/` uses the processed 9505 data to train Bayesian Hidden Markov Models (HMM) on streamflow patterns from the 9505 datasets and generate synthetic streamflow ensembles. These synthetic datasets capture the statistical properties of the 9505 flows while providing multiple realizations for uncertainty analysis and extreme event analysis.

| Script name               | Description                                                                                                                                                        |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| i_train_hmm_models.py     | This script trains a Bayesian Hidden Markov Model on the 9505 data for each basin and ensemble filter combination. It loads historical data to generate prior configurations, and fits HMM models to the filtered 9505 data with diagnostic plotting capabilities. |
| ii_generate_synthetic_streamflow.py | This script loads trained HMM models and generates synthetic streamflow ensembles. It can generate multiple ensemble members and saves the results in NetCDF format for further analysis. |
| iii_explore_streamflow.py | This script loads synthetic streamflow data and produces exploratory plots to visualize the generated ensemble characteristics and compare with historical data. |

### Reproduce WRAP simulations
The workflow in `III_WRAP_Execution/` executes the Water Rights Analysis Package (WRAP) using the synthetic streamflow data to simulate water management scenarios and generate corresponding water management outputs, including both diversion and reservior related variables. This workflow is computationally intensive and requires significant computational resources such as those provided by an HPC environment.

| Script name                    | Description                                                                                                                                                          |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| i_execute_wrap.py              | This script executes WRAP simulations using synthetic streamflow data. It processes multiple ensemble members and filter-basin combinations, running WRAP simulations in parallel and saving the results to NetCDF files. |
| ii_process_diversions_reservoirs.py | This script processes diversions and reservoirs CSV files from WRAP outputs and appends them to the synthetic data NetCDF file. It organizes the data by variable type and ensemble member. |

### Finalize and validate datasets
The workflow in `IV_Finalize_Dataset/` validates the integrity of the complete datasets and optimizes the NetCDF files for distribution.

| Script name                    | Description                                                                                                                                                          |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| i_validate_dataset.py          | This script validates the final NetCDF datasets produced by the workflow for each filter-basin combination. It checks data structure, dimensions, and content to ensure the datasets are complete and properly formatted. |
| ii_optimize_and_finalize.py    | This script optimizes NetCDF files by converting data types to smaller precision, adding internal compression with zlib, and creating a final archive for distribution. Reports size reduction statistics. |

### Computational Requirements
The WRAP execution workflow is computationally expensive, as it involves running the WRAP simulation executable multiple times to create all datasets. There is code for multiprocessing, but a single run of WRAP utilizes a large amount of memory so RAM will be a limiting factor. With 32GB of RAM, 4 processes were able to be concurrently run and took 2-3 days to complete on an Intel i7 processor. With lower RAM and fewer concurrent processes, this could take much longer. Additionally, the code for executing WRAP is designed to run on a Linux operating system using the Windows emulation software [Wine](https://www.winehq.org/). Running the script on Windows or Mac would require edits to the code.

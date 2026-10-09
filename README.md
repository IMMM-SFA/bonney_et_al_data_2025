# bonney_et_al_data_2025

Kirk Bonney<sup>1\*</sup>, Nicole D. Jackson<sup>1</sup>, Stephen Ferencz<sup>2</sup>, Thushara Gunda<sup>1</sup>, and Raquel Valdez<sup>1\*</sup>

<sup>1 </sup> Sandia National Laboratories, Albuquerque, NM, USA
<sup>2 </sup> Pacific Northwest National Laboratory, Richland, WA, USA

\* corresponding author: klbonne@sandia.gov

## Overview
This metarepo contains the utilities and scripts used to generate a dataset of synthetic streamflow realizations and the corresponding water management outputs from the Water Rights Analysis Package (WRAP) for the Colorado, Sabine and Trinity river basins in Texas. The purpose of this repository is to provide the means to reproduce the dataset and to document the process by which it was generated.

## Code reference
Bonney, K., Jackson, N. D., Ferencz, S., Bracken, C., Gunda, T., & Valdez, R. (2026). bonney_et_al_data_2025 (2.0.0). Zenodo. https://doi.org/10.5281/zenodo.21268463 (DOI to be updated when the 2.0.0 release is minted)

## Data reference
| Dataset                                                                          | Link                                                                                          | DOI              |
|----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|------------------|
| Synthetic Streamflow Datasets to Support Emulation of Water Allocations via LSTM | https://data.msdlive.org/records/cfj8d-xsb13                                                  | 10.57931/2441443 |
| Water Availability Model for the Colorado River Basin                            | https://www.tceq.texas.gov/permitting/water_rights/wr_technical-resources/wam.html            | n/a              |
| Water Rights for the Colorado River Basin                                        | https://tceq.maps.arcgis.com/apps/webappviewer/index.html?id=44adc80d90b749cb85cf39e04027dbdc | n/a              |

## Repository layout
| Directory    | Contents                                                                                                                                                      |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `workflow/`  | The end-to-end pipeline, in four stages. |
| `toolkit/`   | The Python package the workflow scripts import: HMM model and bias correction, streamflow disaggregation, WRAP file I/O and execution, NetCDF writers.          |
| `data/`      | Inputs brought in from the accompanying data archive (configuration files, WAM files, the WRAP executable, geospatial data). |
| `outputs/`   | Everything the workflow produces, stage by stage.                                                                                    |
| `tests/`     | Unit tests for the toolkit and a WRAP regression suite (the latter needs Wine and `SIM.exe`).                                                                                    |

## Reproduce this work
Clone this repository (`git clone https://github.com/IMMM-SFA/bonney_et_al_data_2025.git`) and install the `toolkit` package into a Python 3.11 environment. The lockfile reproduces the environment the dataset was built with:

```bash
uv sync --python 3.11        # or: pip install -e .
```

Copy the `data/` folder from the accompanying [MSD-Live archive](https://data.msdlive.org/records/7axm9-yys69) to the top level of the repository. WRAP is a Windows executable and is run through [Wine](https://www.winehq.org/); the code is designed to run on a Linux-based HPC system calls `wine64` by default, and the `WINE_CMD` environment variable overrides that if your Wine installation exposes a different binary name.

The work is reproduced by running the scripts in `workflow/` in order. There are four stage directories, and within each stage the scripts are prefixed with roman numerals giving their order:

| Directory                       | Description                                                                                                      |
|---------------------------------|------------------------------------------------------------------------------------------------------------------|
| `I_9505_Data_Preparation/`      | Downloads the DOE 9505 streamflow projections, associates WRAP control points with river reaches, and builds the HMM training data. |
| `II_HMM_Streamflow_Generation/` | Trains a Bayesian Hidden Markov Model per basin and generates the synthetic streamflow ensemble.                   |
| `III_WRAP_Execution/`           | Runs every realization through WRAP and assembles the diversion and reservoir outputs into the dataset.            |
| `IV_Finalize_Dataset/`          | Validates, compresses and packages the datasets into the distribution archive.                                     |

Every script has two header sections. `### Settings ###` holds the experiment parameters a user may want to change (number of realizations, proportion vector source, compression level, and so on). `### Path Configuration ###` holds the input and output locations; these are derived from the repository layout and should not need editing as long as `data/` is in place. Basin definitions, the 9505 ensemble subsets, random seeds and variable metadata are read from the JSON files in `data/configs/`. Most scripts accept `--filter` and `--basin` arguments to run a single ensemble subset or basin; with no arguments they run the `All Models` subset for all three basins. Implementation detail lives in the `toolkit` package rather than in the scripts.

### Stage I: 9505 data preparation
| Script name                                | Description                                                                                                                                                     |
|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `i_associate_pcp_and_reaches.py`           | Associates each WRAP primary control point with the nearest NHD river reach using the shapefiles in `data/geospatial/`, writing `pcp_to_reach_mapping.csv`. This mapping defines the set of 9505 reaches the rest of the workflow needs. |
| `ii_download_data.py`                      | Downloads the raw 9505 NetCDF files from HydroSource2, scoped to the HUC8 subregions that contain a mapped reach.                                                 |
| `iii_subset_data_to_reaches.py`            | Extracts the mapped reaches from each downloaded file.                                                                                                          |
| `iv_combine_nc_files_and_convert_units.py` | Combines the subsets into one file per 9505 period, tagged by ensemble member, and converts flows from cubic feet per second to acre-feet per month.             |
| `v_explore_9505_data.py`                   | Diagnostic: annual outlet flow distribution of the processed 9505 members against the historical WRAP record, per basin and subset.                            |

### Stage II: HMM training and synthetic streamflow generation
| Script name                           | Description                                                                                                                                                                                                 |
|---------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `i_train_hmm_models.py`               | Fits a two-state Bayesian HMM to annual outlet flow from the 9505 members of each ensemble subset, using priors derived from the historical record, and saves the posterior.                                 |
| `ii_generate_synthetic_streamflow.py` | Samples annual outlet trajectories from the fitted model, bias-corrects them to the historical record by quantile mapping, disaggregates them to monthly flows at every control point using historical analog years, and writes the ensemble to NetCDF. Settings choose the number of realizations, the disaggregation stencil source and whether to apply bias correction. |
| `iii_explore_streamflow.py`           | Diagnostic: MCMC convergence and HMM state plots for each saved model, plus correlation, annual time series and drought metric plots of the synthetic ensemble against the historical record.                |

### Stage III: WRAP execution
| Script name                           | Description                                                                                                                                                                                                                   |
|---------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `i_assign_reservoir_anchors.py`       | One-time configuration step. For each reservoir evaporation site, finds the control point whose historical flow best explains its historical net evaporation and records it as that site's anchor in `data/configs/basins.json`. This is the one place the workflow writes into `data/`; the shipped `basins.json` already contains the result. |
| `ii_execute_wrap.py`                  | Writes each realization to a WRAP `.FLO` file, generates a matching synthetic net-evaporation `.EVA` file from the anchors, runs WRAP in parallel worker slots, and extracts diversion and reservoir outputs to CSV.            |
| `iii_process_diversions_reservoirs.py`| Combines the per-realization CSVs into the diversion and reservoir variables, attaches water-right sector and priority metadata parsed from the WAM, and appends everything to the streamflow NetCDF.                          |
| `iv_explore_wrap_outputs.py`          | Diagnostic: correlation, annual time series and climatology plots of shortage ratio and reservoir net evaporation across realizations, against a single WRAP run of the historical record.                                      |

### Stage IV: dataset finalization
| Script name                     | Description                                                                                                                                                       |
|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `i_validate_dataset.py`         | Checks every expected variable and coordinate is present in each dataset with the right dimensions.                                                               |
| `ii_optimize_and_finalize.py`   | Converts to float32, applies zlib compression, and assembles the distribution archive: one folder per basin, a copy of `data/`, and the archive README (`workflow/MSD-README.md`). |
| `iii_explore_final_dataset.py`  | Diagnostic: streamflow plots read back from the compressed archive files, and per-basin summary tables of streamflow and WRAP statistics.                         |

### Computational requirements
Stage II is single-process and takes about twenty minutes per basin for 2,000 realizations. The three basins can be run concurrently with `--basin`.

Stage III is the expensive stage. One WRAP run takes 45 to 85 seconds depending on the basin, and the script runs them in parallel worker slots. The worker count is read from the `WRAP_NUM_PROCESSES` environment variable (or Slurm's `--cpus-per-task`), defaulting to 3. Each worker needs roughly one gigabyte of scratch space for WRAP's output file, which is kept on `/dev/shm` when that exists so the writes stay in memory, and a few gigabytes of RAM to parse it. The 2,000-realization dataset for all three basins took a few hours with 32 workers on a large shared node and produced about 290 GB of uncompressed output under `outputs/wrap_results/`.

Stage IV loads each dataset into memory to convert and compress it, so it needs RAM on the order of the uncompressed dataset size; the 2,000-realization run was finalized on a node with over a terabyte of memory. The compressed archive is about 22 GB.

WRAP is executed through Wine on Linux. Running on Windows or macOS would require changes to `toolkit/wrap/execution_slot.py`.

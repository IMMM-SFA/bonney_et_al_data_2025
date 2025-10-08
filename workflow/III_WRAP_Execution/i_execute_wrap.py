"""
This script executes WRAP using the synthetic streamflow data and saves selected outputs to the netcdf file.
"""

import os
import multiprocessing
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
import xarray as xr
from toolkit import repo_data_path, outputs_path
from toolkit.wrap.io import flo_to_df
from toolkit.data.io import load_netcdf_format
from toolkit.wrap.wraputils import clean_folders, split_into_sublists, fix_cols
from toolkit.wrap.wraputils import wrap_pipeline, process_ensemble_member


### Settings ###
# Use a conservative number of processes to avoid system freeze
# WRAP simulations are resource-intensive
num_processes = 4  # Use at most 2 processes or half your CPU cores

### Path Configuration ###
WRAP_EXEC_PATH = Path(repo_data_path) / "WRAP" / "wrap_execution_directories"
WRAP_SIM_PATH = WRAP_EXEC_PATH / "SIM.exe"

metadata_path = repo_data_path / "configs" / "wrap_variable_metadata.json"
basins_path = repo_data_path / "configs" / "basins.json"
ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters_basic.json"

### Functions ###
# None

### Main ###

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Execute WRAP simulations for specific filter-basin combinations')
    parser.add_argument('--filter', help='Filter name to process (e.g., basic, cooler, hotter)')
    parser.add_argument('--basin', help='Basin name to process (e.g., Colorado, Trinity, Brazos)')
    args = parser.parse_args()

    # Load basin configuration and variable metadata
    with open(basins_path, "r") as f:
        BASINS = json.load(f)
        
    with open(ensemble_filters_path, "r") as f:
        ENSEMBLE_CONFIG = json.load(f)

    # Load variable metadata
    with open(metadata_path, 'r') as f:
        VARIABLE_METADATA = json.load(f)

    # Filter processing based on arguments
    if args.filter:
        if args.filter not in ENSEMBLE_CONFIG:
            print(f"Error: Filter '{args.filter}' not found in configuration")
            return
        filter_items = [(args.filter, ENSEMBLE_CONFIG[args.filter])]
    else:
        filter_items = ENSEMBLE_CONFIG.items()
    
    if args.basin:
        if args.basin not in BASINS:
            print(f"Error: Basin '{args.basin}' not found in configuration")
            return
        basins = {args.basin: BASINS[args.basin]}
    else:
        basins = BASINS

    # Process selected combinations
    for filter_name, filter_set in filter_items:
        print(f"Processing filter: {filter_name}")
        
        for basin_name, basin in basins.items():
            print(f"  Executing WRAP for basin: {basin_name}")
            gage_name = basin["gage_name"]
            
            # Initialize paths
            flo_file = Path(repo_data_path) / basin["flo_file"]
            dat_file = flo_file.with_suffix(".DAT")
            base_name = flo_file.stem
            synthetic_data_path = Path("outputs") / "bayesian_hmm" / f"{gage_name}_{filter_name}_model" /f"{basin_name.lower()}_results" / f"{basin_name.lower().replace(' ', '_')}_{filter_name}_synthetic_streamflow.nc"
            synthetic_flo_output_path = Path("outputs") / "wrap_results" / basin_name / "synthetic_flos"
            diversions_csvs_path = Path("outputs") / "wrap_results" / basin_name / "diversions"
            reservoirs_csvs_path = Path("outputs") / "wrap_results" / basin_name / "reservoirs"
            out_files_path = Path("outputs") / "wrap_results" / basin_name / "out_files"
            out_zip_path = Path("outputs") / "wrap_results" / basin_name / "wrap_results.zip"

            # ensure necessary directories exist
            for directory_path in [synthetic_flo_output_path, diversions_csvs_path, reservoirs_csvs_path, out_files_path, out_zip_path]:
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)

            # clean out folders from previous runs
            # clean_folders(WRAP_EXEC_PATH, shortage_csvs_path)
            # clean_folders(WRAP_EXEC_PATH, None, None)
            clean_folders(WRAP_EXEC_PATH, diversions_csvs_path, reservoirs_csvs_path, synthetic_flo_output_path, out_files_path)

            # Check is synthetic flo folder is empty
            if len(os.listdir(synthetic_flo_output_path)) == 0:
                # Load synthetic streamflow data

                synthetic_data_dict = load_netcdf_format(synthetic_data_path)
                streamflow = synthetic_data_dict["streamflow"]
                streamflow_index = synthetic_data_dict["streamflow_index"] 
                streamflow_columns = synthetic_data_dict["streamflow_columns"]
                n_ensembles, n_months, n_sites = streamflow.shape
                
                # Load original .FLO as DataFrame to get columns
                flo_df = flo_to_df(str(flo_file))
                flo_columns = flo_df.columns
                flo_index = flo_df.index
                
                    # Prepare arguments for each ensemble member
                ensemble_args = []
                for ens in range(n_ensembles):
                    args = (ens, streamflow, streamflow_index, streamflow_columns, basin_name, flo_df, synthetic_flo_output_path)
                    ensemble_args.append(args)
                
                # Create and start processes
                with multiprocessing.Pool(processes=num_processes) as pool:
                    results = pool.map(process_ensemble_member, ensemble_args)
                
            ## Run wrap pipeline with multiprocessing ##
            # For this code to run correctly, `wrap_execution_folder` needs to point to a directory
            # that contains multiple subdirectories named "execution_folder_i"
            # where i ranges from 0 to the max number of processes you want to run.
            # One such subdirectory is provided in the repo and can be copied as needed.
            # Each of these directories should contain the following files 
            # which are required to run wrap:
            # 
            # C3.DAT
            # C3.DIS
            # C3.EVA
            # C3.FAD
            # C3.HIS

            # obtain list of flo files and split into sublists based on the number of processes
            if len(list(os.listdir(diversions_csvs_path))) == 0 or len(list(os.listdir(reservoirs_csvs_path))) == 0:
                flo_files = os.listdir(synthetic_flo_output_path)
                flo_files.sort()
                sub_lists= split_into_sublists(flo_files, num_processes)

                # run wrap pipeline across multiple processes
                # wrap_pipeline(sub_lists[0], WRAP_EXEC_PATH / "execution_folder_0", WRAP_SIM_PATH, diversions_csvs_path, reservoirs_csvs_path, out_zip_path, synthetic_flo_output_path, out_files_path)
                processes = []
                for process_id, flo_file_list in enumerate(sub_lists):
                    process_wrap_execution_folder = WRAP_EXEC_PATH / f"execution_folder_{process_id}"
                    process = multiprocessing.Process(
                        target=wrap_pipeline, 
                        args=(flo_file_list, process_wrap_execution_folder, WRAP_SIM_PATH, 
                            diversions_csvs_path, reservoirs_csvs_path, out_zip_path, synthetic_flo_output_path, out_files_path)
                    )
                    processes.append(process)
                    process.start()

                for process in processes:
                    process.join()

if __name__ == "__main__":
    main()

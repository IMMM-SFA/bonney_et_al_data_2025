"""
This script executes WRAP using the synthetic streamflow data and saves selected outputs to the netcdf file.
"""

import os
import multiprocessing
from pathlib import Path
import json
import argparse
from toolkit import repo_data_path, outputs_path
from toolkit.wrap.io import flo_to_df
from toolkit.data.io import load_netcdf_format
from toolkit.wrap.execution_slot import WRAPExecutionSlot
from toolkit.wrap.wraputils import clean_folders, split_into_sublists
from toolkit.wrap.wraputils import wrap_pipeline, process_ensemble_member


### Settings ###
# Use a conservative number of processes to avoid system freeze
# WRAP simulations are resource-intensive
num_processes = 4  # Use at most 2 processes or half your CPU cores

### Path Configuration ###
WRAP_EXEC_PATH = Path(repo_data_path) / "WRAP" / "wrap_execution_directories"
WRAP_SIM_PATH = WRAP_EXEC_PATH / ".." / "SIM.exe"

basins_path = repo_data_path / "configs" / "basins.json"
ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters.json"

### Functions ###
# None

### Main ###

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Execute WRAP simulations for specific filter-basin combinations')
    parser.add_argument('--filter', help='Filter name to process (e.g., basic, cooler, hotter)')
    parser.add_argument('--basin', help='Basin name to process (e.g., Colorado, Trinity, Brazos)')
    args = parser.parse_args()

    with open(basins_path, "r") as f:
        BASINS = json.load(f)

    with open(ensemble_filters_path, "r") as f:
        ENSEMBLE_CONFIG = json.load(f)

    # Filter processing based on arguments
    if args.filter:
        filter_sets = [fs for fs in ENSEMBLE_CONFIG if fs["name"] == args.filter]
        if not filter_sets:
            print(f"Error: Filter '{args.filter}' not found in configuration")
            return
    else:
        filter_sets = ENSEMBLE_CONFIG

    if args.basin:
        if args.basin not in BASINS:
            print(f"Error: Basin '{args.basin}' not found in configuration")
            return
        basins = {args.basin: BASINS[args.basin]}
    else:
        basins = BASINS

    # Process selected combinations
    for filter_set in filter_sets:
        filter_name = filter_set["name"]
        print(f"Processing filter: {filter_name}")

        for basin_name, basin in basins.items():
            print(f"  Executing WRAP for basin: {basin_name}")

            # Initialize paths
            flo_file = Path(repo_data_path) / basin["flo_file"]
            base_name = flo_file.stem
            synthetic_data_path = outputs_path / "bayesian_hmm" / f"{filter_name}" / f"{basin_name.lower()}" / f"{filter_name}_{basin_name.lower()}_synthetic_dataset.nc"
            synthetic_flo_output_path = outputs_path / "wrap_results" / filter_name / basin_name / "synthetic_flos"
            diversions_csvs_path = outputs_path / "wrap_results" / filter_name / basin_name / "diversions"
            reservoirs_csvs_path = outputs_path / "wrap_results" / filter_name / basin_name / "reservoirs"

            # ensure necessary directories exist
            for directory_path in [synthetic_flo_output_path, diversions_csvs_path, reservoirs_csvs_path]:
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)

            # Reset execution slots and clean output directories from previous runs
            slots = [
                WRAPExecutionSlot(WRAP_EXEC_PATH / f"execution_folder_{i}", flo_file.parent, WRAP_SIM_PATH, base_name)
                for i in range(num_processes)
            ]
            for slot in slots:
                slot.teardown()
                slot.setup()
            clean_folders(diversions_csvs_path, reservoirs_csvs_path, synthetic_flo_output_path)

            # Check if synthetic flo folder is empty
            if len(os.listdir(synthetic_flo_output_path)) == 0:
                # Load synthetic streamflow data
                synthetic_data_dict = load_netcdf_format(synthetic_data_path)
                streamflow = synthetic_data_dict["streamflow"]
                streamflow_index = synthetic_data_dict["streamflow_index"]
                streamflow_columns = synthetic_data_dict["streamflow_columns"]
                n_ensembles, n_months, n_sites = streamflow.shape

                # Load original .FLO as DataFrame to get columns
                flo_df = flo_to_df(str(flo_file))

                # Prepare arguments for each ensemble member
                ensemble_args = []
                for ens in range(n_ensembles):
                    ens_args = (ens, streamflow, streamflow_index, streamflow_columns, basin, flo_df, synthetic_flo_output_path)
                    ensemble_args.append(ens_args)

                # Create and start processes
                with multiprocessing.Pool(processes=num_processes) as pool:
                    results = pool.map(process_ensemble_member, ensemble_args)

            ## Run wrap pipeline with multiprocessing ##
            if len(list(os.listdir(diversions_csvs_path))) == 0 or len(list(os.listdir(reservoirs_csvs_path))) == 0:
                flo_files = os.listdir(synthetic_flo_output_path)
                flo_files.sort()
                sub_lists = split_into_sublists(flo_files, num_processes)

                processes = []
                for process_id, flo_file_list in enumerate(sub_lists):
                    process = multiprocessing.Process(
                        target=wrap_pipeline,
                        args=(slots[process_id], flo_file_list, diversions_csvs_path, reservoirs_csvs_path, synthetic_flo_output_path)
                    )
                    processes.append(process)
                    process.start()

                for process in processes:
                    process.join()

if __name__ == "__main__":
    main()

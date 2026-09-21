"""
This script executes WRAP using the synthetic streamflow data and saves selected outputs to the netcdf file.
"""

import os
import multiprocessing
from pathlib import Path
import json
from toolkit import repo_data_path, outputs_path
from toolkit.wrap.io import flo_to_df, evp_to_df
from toolkit.data.io import load_netcdf_format
from toolkit.wrap.execution_slot import LocalWRAPExecutionSlot
from toolkit.wrap.wraputils import clean_folders, split_into_sublists
from toolkit.wrap.wraputils import wrap_pipeline, process_ensemble_member
from toolkit.utils.workflow_cli import parse_filter_basin_args, select_filter_sets_and_basins
from toolkit.paths import synthetic_dataset_path


### Settings ###
# Use a conservative number of processes to avoid system freeze
# WRAP simulations are resource-intensive
num_processes = 3

N_ENSEMBLES = 100  # Number of realizations to run through WRAP; None to run all available

DAT_SUFFIX = "_initial_storage_median_historical.dat"

### Path Configuration ###
# WRAP scratch: tmpfs when available (~1 GB RAM per process), else outputs/
_TMPFS = Path("/dev/shm")
WRAP_EXEC_PATH = (_TMPFS / "wrap_exec") if _TMPFS.is_dir() else (outputs_path / "wrap_exec")
WRAP_SIM_PATH = Path(repo_data_path) / "WRAP" / "SIM.exe"

basins_path = repo_data_path / "configs" / "basins.json"
ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters.json"

### Functions ###
# None

### Main ###

def main():
    print(f"Settings: num_processes={num_processes}, N_ENSEMBLES={N_ENSEMBLES}, "
          f"DAT_SUFFIX={DAT_SUFFIX!r}")

    # Parse command line arguments
    args = parse_filter_basin_args('Execute WRAP simulations for specific filter-basin combinations')

    with open(basins_path, "r") as f:
        BASINS = json.load(f)

    with open(ensemble_filters_path, "r") as f:
        ENSEMBLE_CONFIG = json.load(f)

    filter_sets, basins = select_filter_sets_and_basins(BASINS, ENSEMBLE_CONFIG, args.filter, args.basin)

    # Process selected combinations
    for filter_set in filter_sets:
        filter_name = filter_set["name"]
        print(f"Processing filter: {filter_name}")

        for basin_name, basin in basins.items():
            print(f"  Executing WRAP for basin: {basin_name}")

            # Initialize paths
            flo_file = Path(repo_data_path) / basin["flo_file"]
            base_name = flo_file.stem
            synthetic_data_path = synthetic_dataset_path(filter_name, basin_name)
            synthetic_flo_output_path = outputs_path / "wrap_results" / filter_name / basin_name / "synthetic_flos"
            diversions_csvs_path = outputs_path / "wrap_results" / filter_name / basin_name / "diversions"
            reservoirs_csvs_path = outputs_path / "wrap_results" / filter_name / basin_name / "reservoirs"

            # ensure necessary directories exist
            for directory_path in [synthetic_flo_output_path, diversions_csvs_path, reservoirs_csvs_path]:
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)

            # Reset execution slots and clean output directories from previous runs
            slots = [
                LocalWRAPExecutionSlot(WRAP_EXEC_PATH / f"execution_folder_{i}", flo_file.parent, WRAP_SIM_PATH, base_name, dat_suffix=DAT_SUFFIX)
                for i in range(num_processes)
            ]
            for slot in slots:
                slot.teardown()
                slot.setup()
            clean_folders(diversions_csvs_path, reservoirs_csvs_path, synthetic_flo_output_path)

            # Load original .FLO and .EVA as DataFrames (used for both synthetic FLO
            # generation and, further down, per-realization synthetic EVA generation)
            historical_flow_df = flo_to_df(str(flo_file))
            eva_file = next(f for f in flo_file.parent.iterdir() if f.suffix.lower() == ".eva")
            historical_eva_df = evp_to_df(str(eva_file))

            # Check if synthetic flo folder is empty
            if len(os.listdir(synthetic_flo_output_path)) == 0:
                # Load synthetic streamflow data
                synthetic_data_dict = load_netcdf_format(synthetic_data_path)
                streamflow = synthetic_data_dict["streamflow"][:N_ENSEMBLES]
                streamflow_index = synthetic_data_dict["streamflow_index"]
                streamflow_columns = synthetic_data_dict["streamflow_columns"]
                n_ensembles, n_months, n_sites = streamflow.shape

                # Prepare arguments for each ensemble member
                ensemble_args = []
                for ens in range(n_ensembles):
                    ens_args = (ens, streamflow, streamflow_index, streamflow_columns, synthetic_flo_output_path)
                    ensemble_args.append(ens_args)

                # Create and start processes
                with multiprocessing.Pool(processes=num_processes) as pool:
                    pool.map(process_ensemble_member, ensemble_args)

            ## Run wrap pipeline with multiprocessing ##
            if len(list(os.listdir(diversions_csvs_path))) == 0 or len(list(os.listdir(reservoirs_csvs_path))) == 0:
                flo_files = os.listdir(synthetic_flo_output_path)
                flo_files.sort()
                sub_lists = split_into_sublists(flo_files, num_processes)

                processes = []
                for process_id, flo_file_list in enumerate(sub_lists):
                    process = multiprocessing.Process(
                        target=wrap_pipeline,
                        args=(
                            slots[process_id], flo_file_list, diversions_csvs_path, reservoirs_csvs_path,
                            synthetic_flo_output_path, historical_eva_df, historical_flow_df, basin,
                        )
                    )
                    processes.append(process)
                    process.start()

                for process in processes:
                    process.join()

            for slot in slots:
                slot.teardown()

if __name__ == "__main__":
    main()

"""
This script processes diversions and reservoirs CSV files and combines them with the
synthetic streamflow data into a single WRAP-augmented NetCDF file.
"""


import os
import multiprocessing
import numpy as np
import pandas as pd
import json
import xarray as xr
from toolkit import repo_data_path, outputs_path
from toolkit.utils.workflow_cli import parse_filter_basin_args, select_filter_sets_and_basins
from toolkit.paths import synthetic_dataset_path, wrap_augmented_dataset_path


### Settings ###
# Use a conservative number of processes to avoid system freeze
# Processing CSV files and NetCDF operations are resource-intensive
num_processes = 4  # Use at most 4 processes or half your CPU cores

### Path Configuration ###
metadata_path = repo_data_path / "configs" / "wrap_variable_metadata.json"
basins_path = repo_data_path / "configs" / "basins.json"
ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters.json"

### Functions ###
def process_filter_basin_combination(args):
    """
    Worker function to process a single filter-basin combination.
    
    Parameters
    ----------
    args : tuple
        Contains (filter_name, basin_name, basin, variable_metadata)

    Returns
    -------
    str
        Success message
    """
    filter_name, basin_name, basin, variable_metadata = args

    print(f"  Processing basin: {basin_name} with filter: {filter_name}")

    # Initialize paths
    synthetic_data_path = synthetic_dataset_path(filter_name, basin_name)
    output_path = wrap_augmented_dataset_path(filter_name, basin_name)
    diversions_csvs_path = outputs_path / "wrap_results" / filter_name / basin_name / "diversions"
    reservoirs_csvs_path = outputs_path / "wrap_results" / filter_name / basin_name / "reservoirs"

    # Process diversions and reservoirs
    process_diversions_and_reservoirs(synthetic_data_path, output_path, diversions_csvs_path, reservoirs_csvs_path, variable_metadata)

    return f"Successfully processed {filter_name} - {basin_name}"

def _group_files_by_variable(csvs_path):
    """Groups a directory of `synthflow_<N>_<variable>.csv` files by variable name,
    each list sorted by ensemble number N."""
    file_list = list(os.listdir(csvs_path))
    file_list.sort(key=lambda string: int(string.split("_")[1]))

    file_groups = {}
    for csv_file in file_list:
        if csv_file.endswith('.csv'):
            parts = csv_file.split('_')
            if len(parts) >= 3:
                variable_name = '_'.join(parts[2:]).replace('.csv', '')
                file_groups.setdefault(variable_name, []).append(csv_file)
    return file_groups

def _build_variable_dataarrays(csvs_path, file_groups, dim_name, id_coord,
                                realization_coords, time_step_coords, metadata_section):
    """Loads each variable's per-realization CSVs into one DataArray per variable,
    keyed by variable name."""
    dataarrays = {}
    for variable_name, file_list in file_groups.items():
        print(f"Processing {dim_name} {variable_name} with {len(file_list)} files...")

        first_file = file_list[0]
        example_df = pd.read_csv(csvs_path / first_file, index_col=0)
        n_time, n_ids = example_df.shape

        variable_data = np.zeros((len(file_list), n_time, n_ids))
        for i, csv_file in enumerate(file_list):
            df = pd.read_csv(csvs_path / csv_file, index_col=0)
            variable_data[i, :, :] = df.values

        if variable_name in metadata_section:
            metadata = metadata_section[variable_name]
        else:
            metadata = {
                'long_name': variable_name.replace('_', ' ').title(),
                'units': 'unknown',
                'description': f'{dim_name.title()} {variable_name.replace("_", " ")} data from WRAP model simulation',
                'standard_name': variable_name
            }

        dataarrays[variable_name] = xr.DataArray(
            variable_data,
            dims=['realization', 'time_step', id_coord.dims[0]],
            coords={
                'realization': realization_coords,
                'time_step': time_step_coords,
                id_coord.dims[0]: id_coord,
            },
            attrs={
                'long_name': metadata['long_name'],
                'units': metadata['units'],
                'description': metadata['description'],
            }
        )
    return dataarrays

def process_diversions_and_reservoirs(synthetic_data_path, output_path, diversions_csvs_path, reservoirs_csvs_path, variable_metadata):
    """
    Combine diversions/reservoirs CSV files with the synthetic streamflow data into
    a single WRAP-augmented NetCDF, written fresh to output_path.

    Parameters
    ----------
    synthetic_data_path : Path
        Path to the Stage II synthetic streamflow NetCDF (read-only source).
    output_path : Path
        Path to write the combined NetCDF to.
    diversions_csvs_path : Path
        Path to directory containing diversions CSV files
    reservoirs_csvs_path : Path
        Path to directory containing reservoirs CSV files
    """

    print("Processing diversions data...")
    diversions_file_groups = _group_files_by_variable(diversions_csvs_path)
    print(f"Found {len(diversions_file_groups)} diversion variable types: {list(diversions_file_groups.keys())}")

    print("Processing reservoirs data...")
    reservoirs_file_groups = _group_files_by_variable(reservoirs_csvs_path)
    print(f"Found {len(reservoirs_file_groups)} reservoir variable types: {list(reservoirs_file_groups.keys())}")

    # However many realizations actually got CSVs is however many WRAP was run for.
    n_ensembles = len(next(iter(diversions_file_groups.values()), next(iter(reservoirs_file_groups.values()), [])))

    with xr.open_dataset(synthetic_data_path) as ds:
        combined_ds = ds.isel(realization=slice(0, n_ensembles)).load()

    if 'n_realizations' in combined_ds.attrs:
        combined_ds.attrs['n_realizations'] = n_ensembles

    realization_coords = combined_ds['realization']
    time_step_coords = combined_ds['time_step']

    if diversions_file_groups:
        first_file = next(iter(diversions_file_groups.values()))[0]
        example_df = pd.read_csv(diversions_csvs_path / first_file, index_col=0)
        right_id_coord = xr.DataArray(
            example_df.columns,
            dims=['right_id'],
            attrs=dict(variable_metadata['coordinate_variables']['right_id']),
        )
        diversion_das = _build_variable_dataarrays(
            diversions_csvs_path, diversions_file_groups, 'diversion', right_id_coord,
            realization_coords, time_step_coords, variable_metadata['diversion'],
        )
        combined_ds = combined_ds.assign(diversion_das)

    if reservoirs_file_groups:
        first_file = next(iter(reservoirs_file_groups.values()))[0]
        example_df = pd.read_csv(reservoirs_csvs_path / first_file, index_col=0)
        reservoir_id_coord = xr.DataArray(
            example_df.columns,
            dims=['reservoir_id'],
            attrs=dict(variable_metadata['coordinate_variables']['reservoir_id']),
        )
        reservoir_das = _build_variable_dataarrays(
            reservoirs_csvs_path, reservoirs_file_groups, 'reservoir', reservoir_id_coord,
            realization_coords, time_step_coords, variable_metadata['reservoir'],
        )
        combined_ds = combined_ds.assign(reservoir_das)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_ds.to_netcdf(output_path)
    print(f"Wrote combined dataset ({n_ensembles} realizations) to {output_path}")

### Main ###

def main():
    # Parse command line arguments
    args = parse_filter_basin_args('Process diversions and reservoirs for specific filter-basin combinations')

    # Load basin configuration and variable metadata
    with open(basins_path, "r") as f:
        BASINS = json.load(f)

    # Load ensemble filters configuration
    with open(ensemble_filters_path, "r") as f:
        ENSEMBLE_CONFIG = json.load(f)

    # Load variable metadata
    with open(metadata_path, 'r') as f:
        variable_metadata = json.load(f)

    filter_sets, basins = select_filter_sets_and_basins(BASINS, ENSEMBLE_CONFIG, args.filter, args.basin)

    # Collect all filter-basin combinations
    all_combinations = []
    for filter_set in filter_sets:
        filter_name = filter_set["name"]
        for basin_name, basin in basins.items():
            all_combinations.append((filter_name, basin_name, basin, variable_metadata))

    print(f"Processing {len(all_combinations)} filter-basin combinations...")

    # Process all combinations in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_filter_basin_combination, all_combinations)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()


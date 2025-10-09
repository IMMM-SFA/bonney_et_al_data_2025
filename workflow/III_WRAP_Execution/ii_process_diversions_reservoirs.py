"""
This script processes diversions and reservoirs CSV files and appends them to the synthetic data NetCDF file.
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


### Settings ###


### Path Configuration ###
WRAP_EXEC_PATH = Path(repo_data_path) / "WRAP" / "wrap_execution_directories"
WRAP_SIM_PATH = WRAP_EXEC_PATH / "SIM.exe"

metadata_path = Path(__file__).parent / "wrap_variable_metadata.json"
basins_path = Path(__file__).parent / "basins.json"

### Functions ###
def process_diversions_and_reservoirs(synthetic_data_path, diversions_csvs_path, reservoirs_csvs_path):
    """
    Process diversions and reservoirs CSV files and append them to the synthetic data NetCDF file.
    
    Parameters
    ----------
    synthetic_data_path : Path
        Path to the synthetic data NetCDF file
    diversions_csvs_path : Path
        Path to directory containing diversions CSV files
    reservoirs_csvs_path : Path
        Path to directory containing reservoirs CSV files
    """
    
    ## Process Diversions CSV files ##
    
    print("Processing diversions data...")
    diversions_file_list = list(os.listdir(diversions_csvs_path))
    diversions_file_list.sort(key=lambda string: int(string.split("_")[1]))
    
    # Group diversion files by variable
    diversions_file_groups = {}
    for csv_file in diversions_file_list:
        if csv_file.endswith('.csv'):
            # Extract ensemble number and variable type from filename
            # Format: ensemble_N_variable.csv
            parts = csv_file.split('_')
            if len(parts) >= 3:
                variable_name = '_'.join(parts[2:]).replace('.csv', '')
                
                if variable_name not in diversions_file_groups:
                    diversions_file_groups[variable_name] = []
                diversions_file_groups[variable_name].append(csv_file)
    
    print(f"Found {len(diversions_file_groups)} diversion variable types: {list(diversions_file_groups.keys())}")
    
    # Load existing NetCDF file and create coordinate variables once
    with xr.open_dataset(synthetic_data_path) as ds:
        # Create right_id coordinate variable (only once)
        if diversions_file_groups:
            first_file = list(diversions_file_groups.values())[0][0]
            example_df = pd.read_csv(diversions_csvs_path / first_file, index_col=0)
            right_id_coord = xr.DataArray(
                example_df.columns,
                dims=['right_id'],
                attrs={
                    'long_name': 'Water Right Identifier',
                    'description': 'Index of water right identifiers',
                    'standard_name': 'right_id',
                }
            )
        
        # Process each diversion variable type
        for variable_name, file_list in diversions_file_groups.items():
            print(f"Processing diversion {variable_name} with {len(file_list)} files...")
            
            # Load first file to get data shape
            first_file = file_list[0]
            example_df = pd.read_csv(diversions_csvs_path / first_file, index_col=0)
            
            # Initialize array for this variable
            n_ensembles = len(file_list)
            n_time = example_df.shape[0]
            n_rights = example_df.shape[1]
            
            variable_data = np.zeros((n_ensembles, n_time, n_rights))
            
            # Load data for each ensemble
            for i, csv_file in enumerate(file_list):
                df = pd.read_csv(diversions_csvs_path / csv_file, index_col=0)
                variable_data[i, :, :] = df.values
            
            # Get metadata for this variable
            if variable_name in VARIABLE_METADATA['diversion']:
                metadata = VARIABLE_METADATA['diversion'][variable_name]
            else:
                # Fallback metadata if variable not found
                metadata = {
                    'long_name': variable_name.replace('_', ' ').title(),
                    'units': 'unknown',
                    'description': f'Diversion {variable_name.replace("_", " ")} data from WRAP model simulation',
                    'standard_name': f'diversion_{variable_name}'
                }
            
            # Create variable data array
            variable_da = xr.DataArray(
                variable_data,
                dims=['ensemble', 'time', 'right_id'],
                coords={
                    'ensemble': ds['ensemble'],
                    'time': ds['time'],
                    'right_id': right_id_coord
                },
                attrs={
                    'long_name': metadata['long_name'],
                    'units': metadata['units'],
                    'description': metadata['description'],
                    'standard_name': metadata['standard_name']
                }
            )
            
            # Create a new dataset with just this variable
            new_ds = xr.Dataset({f'diversion_{variable_name}': variable_da})
            
            # Append to existing NetCDF file using xarray's append functionality
            new_ds.to_netcdf(synthetic_data_path, mode='a')
            print(f"Successfully appended diversion {variable_name} data to {synthetic_data_path}")

    ## Process Reservoirs CSV files ##
    
    print("Processing reservoirs data...")
    reservoirs_file_list = list(os.listdir(reservoirs_csvs_path))
    reservoirs_file_list.sort(key=lambda string: int(string.split("_")[1]))
    
    # Group reservoir files by variable
    reservoirs_file_groups = {}
    for csv_file in reservoirs_file_list:
        if csv_file.endswith('.csv'):
            # Extract ensemble number and variable type from filename
            # Format: ensemble_N_variable.csv
            parts = csv_file.split('_')
            if len(parts) >= 3:
                variable_name = '_'.join(parts[2:]).replace('.csv', '')
                
                if variable_name not in reservoirs_file_groups:
                    reservoirs_file_groups[variable_name] = []
                reservoirs_file_groups[variable_name].append(csv_file)
    
    print(f"Found {len(reservoirs_file_groups)} reservoir variable types: {list(reservoirs_file_groups.keys())}")
    
    # Load existing NetCDF file and create coordinate variables once
    with xr.open_dataset(synthetic_data_path) as ds:
        # Create reservoir_id coordinate variable (only once)
        if reservoirs_file_groups:
            first_file = list(reservoirs_file_groups.values())[0][0]
            example_df = pd.read_csv(reservoirs_csvs_path / first_file, index_col=0)
            reservoir_id_coord = xr.DataArray(
                example_df.columns,
                dims=['reservoir_id'],
                attrs={
                    'long_name': 'Reservoir Identifier',
                    'description': 'Index of reservoir identifiers',
                    'standard_name': 'reservoir_id',
                }
            )
        
        # Process each reservoir variable type
        for variable_name, file_list in reservoirs_file_groups.items():
            print(f"Processing reservoir {variable_name} with {len(file_list)} files...")
            
            # Load first file to get data shape
            first_file = file_list[0]
            example_df = pd.read_csv(reservoirs_csvs_path / first_file, index_col=0)
            
            # Initialize array for this variable
            n_ensembles = len(file_list)
            n_time = example_df.shape[0]
            n_reservoirs = example_df.shape[1]
            
            variable_data = np.zeros((n_ensembles, n_time, n_reservoirs))
            
            # Load data for each ensemble
            for i, csv_file in enumerate(file_list):
                df = pd.read_csv(reservoirs_csvs_path / csv_file, index_col=0)
                variable_data[i, :, :] = df.values
            
            # Get metadata for this variable
            if variable_name in VARIABLE_METADATA['reservoir']:
                metadata = VARIABLE_METADATA['reservoir'][variable_name]
            else:
                # Fallback metadata if variable not found
                metadata = {
                    'long_name': variable_name.replace('_', ' ').title(),
                    'units': 'unknown',
                    'description': f'Reservoir {variable_name.replace("_", " ")} data from WRAP model simulation',
                    'standard_name': f'reservoir_{variable_name}'
                }
            
            # Create variable data array
            variable_da = xr.DataArray(
                variable_data,
                dims=['ensemble', 'time', 'reservoir_id'],
                coords={
                    'ensemble': ds['ensemble'],
                    'time': ds['time'],
                    'reservoir_id': reservoir_id_coord
                },
                attrs={
                    'long_name': metadata['long_name'],
                    'units': metadata['units'],
                    'description': metadata['description'],
                    'standard_name': metadata['standard_name']
                }
            )
            
            # Create a new dataset with just this variable
            new_ds = xr.Dataset({f'reservoir_{variable_name}': variable_da})
            
            # Append to existing NetCDF file using xarray's append functionality
            new_ds.to_netcdf(synthetic_data_path, mode='a')
            print(f"Successfully appended reservoir {variable_name} data to {synthetic_data_path}")

### Main ###

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process diversions and reservoirs for specific filter-basin combinations')
    parser.add_argument('--filter', help='Filter name to process (e.g., basic, cooler, hotter)')
    parser.add_argument('--basin', help='Basin name to process (e.g., Colorado, Trinity, Brazos)')
    args = parser.parse_args()

    # Load basin configuration and variable metadata
    with open(basins_path, "r") as f:
        BASINS = json.load(f)

    # Load ensemble filters configuration
    ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters_basic.json"
    with open(ensemble_filters_path, "r") as f:
        ENSEMBLE_CONFIG = json.load(f)

    # Load variable metadata
    with open(metadata_path, 'r') as f:
        VARIABLE_METADATA = json.load(f)

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
    for filter_set in filter_items:
        filter_name = filter_set["name"]
        print(f"Processing filter: {filter_name}")
        
        for basin_name, basin in basins.items():
            gage_name = basin["gage_name"]
            
            print(f"  Processing basin: {basin_name} with filter: {filter_name}")
            
            # Initialize paths
            synthetic_data_path = outputs_path / "bayesian_hmm" / f"{filter_name}" /f"{basin_name.lower()}" / f"{filter_name}_{basin_name.lower()}_synthetic_streamflow.nc"
            diversions_csvs_path = outputs_path / "wrap_results" / basin_name / "diversions"
            reservoirs_csvs_path = outputs_path / "wrap_results" / basin_name / "reservoirs"
            
            # Process diversions and reservoirs
            process_diversions_and_reservoirs(synthetic_data_path, diversions_csvs_path, reservoirs_csvs_path)

if __name__ == "__main__":
    main()


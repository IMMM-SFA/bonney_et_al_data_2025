"""
This script subsets the 9505 data to the reaches of interest and saves it to a NetCDF file.
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import os
from toolkit import repo_data_path, outputs_path


### Settings ###
# None

### Path Configuration ###
data_root = outputs_path / "9505" / "raw"  # Change to actual data directory
output_root = outputs_path / "9505" / "reach_subset"  # Change this to where you want to save the files
output_root.mkdir(parents=True, exist_ok=True)

desired_reaches = pd.read_csv(repo_data_path / "configs" / "reaches_of_interest.csv")

### Functions ###
def process_huc8_file(nc_file, desired_reaches):
    """Extracts streamflow data for specific reaches from a single HUC8 file.

    Args:
        nc_file: Path to the NetCDF file
        desired_reaches: List of COMIDs to extract

    Returns:
        tuple: (reaches_in_file, time_mn, data_matrix)
            - reaches_in_file: list of COMIDs found in this file (subset of desired_reaches)
            - time_mn: array of time values
            - data_matrix: 2D array (n_reaches, n_time) of streamflow values
    """
    print(f"Processing: {nc_file}")
    try:
        with xr.open_dataset(nc_file) as ds:
            streamflow = ds["RAPID_mn_cfs"]  # (comid, time_mn)
            time_mn = ds["time_mn"].values
            comid = ds["COMID"].values

            # Find which desired reaches are in this file
            mask = np.isin(comid, desired_reaches)
            if not np.any(mask):
                print(f"No desired reaches found in {nc_file}, skipping.")
                return None

            reaches_in_file = comid[mask]
            data_matrix = streamflow[mask, :].values  # shape: (n_reaches, n_time)

            return reaches_in_file, time_mn, data_matrix
    except Exception as e:
        print(f"Error processing {nc_file}: {e}")
        return None

def process_folder(folder, desired_reaches, force_compute=False):
    """Processes all HUC8 files in a folder and combines streamflow data for specific reaches.

    Args:
        folder: Path to the folder containing HUC8 files
        desired_reaches: List of COMIDs to extract
        force_compute: Whether to recompute if output already exists
    """
    folder_name = folder.name
    print(f"Processing folder: {folder_name}")

    file_path = output_root / f"{folder_name}.nc"
    if not force_compute and os.path.exists(file_path):
        print(f"Folder {folder_name} already processed, skipping.")
        return

    nc_files = list(folder.glob("*.nc"))
    if not nc_files:
        print(f"No NetCDF files found in {folder_name}, skipping.")
        return

    # Prepare to collect data for all desired reaches and all times
    all_time = set()
    reach_to_data = {}

    for nc_file in nc_files:
        result = process_huc8_file(nc_file, desired_reaches)
        if result:
            reaches_in_file, time_mn, data_matrix = result
            all_time.update(time_mn)
            for i, reach in enumerate(reaches_in_file):
                if reach not in reach_to_data:
                    reach_to_data[reach] = []
                reach_to_data[reach].append((time_mn, data_matrix[i, :]))

    if not reach_to_data:
        print(f"No valid data found in {folder_name}, skipping.")
        return

    # Sort and index time and reaches
    all_time = np.array(sorted(all_time))
    all_reaches = np.array(sorted(reach_to_data.keys()))

    # Prepare output array: (n_reaches, n_time)
    data_out = np.full((len(all_reaches), len(all_time)), np.nan)

    # Fill output array
    for i, reach in enumerate(all_reaches):
        for time_mn, data in reach_to_data[reach]:
            time_idx = np.searchsorted(all_time, time_mn)
            data_out[i, time_idx] = data

    # Create xarray dataset
    ds_out = xr.Dataset(
        {
            "streamflow": (["reach", "time_mn"], data_out)
        },
        coords={
            "reach": all_reaches,
            "time_mn": all_time
        }
    )

    ds_out.to_netcdf(file_path)
    print(f"Saved: {file_path}")
    """Processes all HUC8 files in a folder and combines streamflow data for specific reaches.
    
    Args:
        folder: Path to the folder containing HUC8 files
        desired_reaches: List of COMIDs to extract
        force_compute: Whether to recompute if output already exists
    """
    folder_name = folder.name  # Use folder name for the output filename
    print(f"Processing folder: {folder_name}")
    
    # check if output already exists
    file_path = output_root / f"{folder_name}.nc"
    if not force_compute and os.path.exists(file_path):
        print(f"Folder {folder_name} already processed, skipping.")
        return

    results = []
    
    nc_files = list(folder.glob("*.nc"))
    
    for nc_file in nc_files:
        result = process_huc8_file(nc_file, desired_reaches)
        if result:
            results.append(result)

    if not results:
        print(f"No valid data found in {folder_name}, skipping.")
        return

    # Combine data from all files
    all_reaches = set()
    for _, _, reach_data in results:
        all_reaches.update(reach_data.keys())
    
    # Ensure time is consistent across all files
    time_values = np.unique(np.concatenate([time for _, time, _ in results]))
    
    # Create arrays for each reach
    reach_arrays = {}
    for reach in all_reaches:
        reach_arrays[reach] = np.full(len(time_values), np.nan)
    
    # Populate the arrays
    for _, time_mn, reach_data in results:
        time_idx = np.searchsorted(time_values, time_mn)
        for reach, data in reach_data.items():
            reach_arrays[reach][time_idx] = data
    
    # Create xarray dataset with each reach as a variable
    ds_out = xr.Dataset(
        {f"reach_{reach}": (["time_mn"], reach_arrays[reach]) 
         for reach in all_reaches},
        coords={"time_mn": time_values}
    )

    # Save to NetCDF
    ds_out.to_netcdf(file_path)
    print(f"Saved: {file_path}")

def process_all_folders(data_root, desired_reaches):
    """Processes each model run folder separately and generates one aggregated NetCDF file per folder."""
    for folder in data_root.iterdir():
        if folder.is_dir():
            process_folder(folder, desired_reaches)

### Main ###

desired_reaches = list(desired_reaches.iloc[:, 1])

# Run processing
process_all_folders(data_root, desired_reaches=desired_reaches)

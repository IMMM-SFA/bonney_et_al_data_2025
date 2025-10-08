"""
This script combines the 9505 data into a single NetCDF file and converts to acre-feet.
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from toolkit import outputs_path
from calendar import monthrange

### Settings ###
# None

### Path Configuration ###
input_folder = outputs_path / "9505" / "reach_subset"
output_folder = outputs_path / "9505" / "reach_subset_combined"

# Create output directory if it doesn't exist
output_folder.mkdir(parents=True, exist_ok=True)

### Functions ###
def extract_metadata_from_filename(filename):
    """Extracts metadata dynamically from NetCDF filenames using underscore-based parsing."""
    filename = Path(filename).stem  # Remove ".nc"
    parts = filename.split("_")

    # Ensure we only process valid filenames with >6 parts
    if len(parts) <= 6:
        raise ValueError(f"Skipping {filename}: Insufficient metadata fields.")

    return {
        "prefix1": parts[0],      # e.g., "PRMS" or "VIC5"
        "prefix2": parts[1],      # e.g., "RAPID"
        "original_filename": filename + ".nc",
        "model": parts[2],        # e.g., "EC-Earth3"
        "scenario": parts[3],     # e.g., "ssp245" or "historical"
        "variant_id": parts[4] if "r" in parts[4] else None,  # e.g., "r1i1p1f1"
        "downscaling": parts[5] if "r" not in parts[5] else parts[6],  # e.g., "DBCCA" or "Daymet2019"
        "rmf": parts[6] if "r" not in parts[5] else parts[7],  # e.g., "Daymet"
        "year_start": parts[-2],  # e.g., "2060"
        "year_end": parts[-1],    # e.g., "2099"
    }

def cfs_to_acre_feet(flow_cfs, date):
    """Convert flow from CFS to acre-feet for a given month."""
    _, days_in_month = monthrange(date.year, date.month)
    seconds_in_month = days_in_month * 24 * 3600
    flow_af = flow_cfs * seconds_in_month / 43560
    return flow_af

def create_datetime_index(start_year, end_year):
    """Create a datetime index for the given year range."""
    return pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="MS")

def load_and_preprocess_nc(file_path):
    """Loads a NetCDF file, extracts metadata, and prepares it for combination."""
    ds = xr.open_dataset(file_path)

    # Extract metadata
    try:
        metadata = extract_metadata_from_filename(file_path)
    except ValueError as e:
        print(e)
        return None

    # Expand dataset along the new `ensemble_id` axis
    ds = ds.expand_dims(dim={"ensemble_id": [metadata["original_filename"]]})

    # Attach metadata as variables (these will persist in NetCDF)
    for key, value in metadata.items():
        ds[key] = (['ensemble_id'], [value])

    return ds

def convert_to_acre_feet(ds):
    """Convert streamflow from CFS to acre-feet."""
    # Create datetime index
    start_year = int(ds.year_start.values[0])
    end_year = int(ds.year_end.values[0])
    datetime_index = create_datetime_index(start_year, end_year)
    
    # Create new dataset for acre-feet values
    ds_af = xr.Dataset()
    
    # Convert each reach variable
    for var in ds.data_vars:
        if var.startswith('reach_'):
            # Get the data
            data = ds[var].values
            
            # Convert to acre-feet
            converted_data = np.zeros_like(data)
            for i, date in enumerate(datetime_index):
                converted_data[:, i] = cfs_to_acre_feet(data[:, i], date)
            
            # Add to new dataset
            ds_af[var] = xr.DataArray(
                converted_data,
                dims=["ensemble_id", "time_mn"],
                coords={"ensemble_id": ds.ensemble_id, "time_mn": datetime_index}
            )
            ds_af[var].attrs["units"] = "acre-feet/month"
    
    # Copy over 1D metadata variables (e.g., model, scenario, etc.)
    for var in ds.data_vars:
        if var not in ds_af and ds[var].dims == ('ensemble_id',):
            ds_af[var] = ds[var]
    
    # Copy over metadata
    ds_af.attrs = ds.attrs
    ds_af.attrs["units"] = "acre-feet/month"
    
    return ds_af

def combine_nc_files_by_time_period(input_folder, output_folder):
    """
    Combines NetCDF files into master datasets per time period and converts to acre-feet.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Collect files by time period
    grouped_files = {}

    for file_path in input_folder.glob("*.nc"):
        try:
            metadata = extract_metadata_from_filename(file_path)
            time_period = f"{metadata['year_start']}_{metadata['year_end']}"

            if time_period not in grouped_files:
                grouped_files[time_period] = []
            grouped_files[time_period].append(file_path)

        except ValueError as e:
            print(f"Skipping file due to format: {file_path} - {e}")

    # Process each time period
    for time_period, files in grouped_files.items():
        print(f"Processing time period: {time_period} with {len(files)} files")

        # Load and preprocess all files
        datasets = [load_and_preprocess_nc(file) for file in files if file]
        datasets = [ds for ds in datasets if ds is not None]  # Remove None values

        if not datasets:
            print(f"No valid datasets for {time_period}. Skipping...")
            continue

        # Align along the time dimension
        combined_ds = xr.concat(datasets, dim="ensemble_id")
        
        # Save the CFS version
        output_file = output_folder / f"master_streamflow_{time_period}.nc"
        combined_ds.to_netcdf(output_file, format='NETCDF4')
        print(f"Saved CFS version: {output_file}")
        
        # Convert to acre-feet and save
        ds_af = convert_to_acre_feet(combined_ds)
        output_file_af = output_folder / f"master_streamflow_{time_period}_af.nc"
        ds_af.to_netcdf(output_file_af, format='NETCDF4')
        print(f"Saved acre-feet version: {output_file_af}")

### Main ###

combine_nc_files_by_time_period(input_folder, output_folder)

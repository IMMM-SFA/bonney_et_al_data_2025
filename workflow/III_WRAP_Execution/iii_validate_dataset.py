"""
WRAP Outputs Dataset Validation

This script validates the final NetCDF datasets produced by the workflow for
each filter-basin combination. It assumes the full workflow has completed and
only checks the NetCDF datasets (no CSV or FLO files).

Validates:
1. Synthetic Streamflow NetCDF with integrated WRAP variables (diversions/reservoirs)
"""

import xarray as xr
import numpy as np
from pathlib import Path
import sys
import json
import argparse

from toolkit import repo_data_path, outputs_path

### Settings ###
# None

### Path Configuration ###
basins_path = repo_data_path / "configs" / "basins.json"
ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters_basic.json"

### Functions ###

def build_synthetic_dataset_path(filter_name: str, basin_name: str) -> Path:
    return outputs_path / "bayesian_hmm" / f"{filter_name}" / f"{basin_name.lower()}" / f"{filter_name}_{basin_name.lower()}_synthetic_dataset.nc"

def validate_synthetic_dataset(file_path: Path) -> dict:
    """Validate synthetic streamflow dataset structure."""
    print(f"  Validating synthetic streamflow: {file_path.name}")
    
    ds = xr.open_dataset(file_path)
    results = {
        "file_path": str(file_path),
        "file_type": "synthetic_dataset",
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check required dimensions according to README
    required_dims = ["ensemble", "time", "site", "year", "parameter"]
    for dim in required_dims:
        if dim not in ds.dims:
            results["errors"].append(f"Missing required dimension: {dim}")
            results["valid"] = False
    
    # Check for right_id dimension (required if shortage data exists)
    has_shortage = "diversion_shortage_ratio" in ds.data_vars
    if has_shortage and "right_id" not in ds.dims:
        results["errors"].append("Missing required dimension: right_id (needed for shortage data)")
        results["valid"] = False
    
    # Check required data variables
    required_streamflow_data_vars = ["streamflow", "annual_states", "hmm_parameters"]
    for var in required_streamflow_data_vars:
        if var not in ds.data_vars:
            results["errors"].append(f"Missing required data variable: {var}")
            results["valid"] = False
    
    required_diversion_data_vars = ["diversion_diversion_or_energy_shortage", 
                                    "diversion_diversion_or_energy_target", 
                                    "diversion_shortage_ratio"]
    for var in required_diversion_data_vars:
        if var not in ds.data_vars:
            results["errors"].append(f"Missing required data variable: {var}")
            results["valid"] = False
    
    required_reservoir_data_vars = [
        "reservoir_reservoir_releases_not_accessible_to_hydroelectric_power_turbines",
        "reservoir_reservoir_storage_capacity",
        "reservoir_reservoir_net_evaporation_precipitation_volume",
        "reservoir_inflows_to_reservoir_from_releases_from_other_reservoirs",
        "reservoir_energy_generated",
        "reservoir_inflows_to_reservoir_from_stream_flow_depletions",
        "reservoir_reservoir_water_surface_elevation",
        "reservoir_reservoir_releases_accessible_to_hydroelectric_power_turbines",]
    for var in required_reservoir_data_vars:
        if var not in ds.data_vars:
            results["errors"].append(f"Missing required data variable: {var}")
            results["valid"] = False
    
    # Validate streamflow variable
    if "streamflow" in ds.data_vars:
        streamflow = ds["streamflow"]
        expected_dims = ["ensemble", "time", "site"]
        if list(streamflow.dims) != expected_dims:
            results["errors"].append(f"Streamflow dimensions {list(streamflow.dims)} don't match expected {expected_dims}")
            results["valid"] = False
        
        # Check units
        if "units" in streamflow.attrs:
            units = streamflow.attrs["units"]
            if "acre-feet" not in units.lower():
                results["warnings"].append(f"Streamflow units may be incorrect: {units}")
        else:
            results["warnings"].append("Streamflow missing units attribute")
        
        # Check other required attributes
        required_attrs = ["long_name", "description", "standard_name"]
        for attr in required_attrs:
            if attr not in streamflow.attrs:
                results["warnings"].append(f"Streamflow missing {attr} attribute")
    
    # Validate shortage variable (if present)
    if "diversion_shortage_ratio" in ds.data_vars:
        shortage = ds["diversion_shortage_ratio"]
        expected_dims = ["ensemble", "time", "right_id"]
        if list(shortage.dims) != expected_dims:
            results["errors"].append(f"Shortage dimensions {list(shortage.dims)} don't match expected {expected_dims}")
            results["valid"] = False
        
        # Check units
        if "units" in shortage.attrs:
            units = shortage.attrs["units"]
            if "ratio" not in units.lower():
                results["warnings"].append(f"Shortage units may be incorrect: {units}")
        else:
            results["warnings"].append("Shortage missing units attribute")
        
        # Check other required attributes
        required_attrs = ["long_name", "description", "standard_name"]
        for attr in required_attrs:
            if attr not in shortage.attrs:
                results["warnings"].append(f"Shortage missing {attr} attribute")
    
    # Validate annual_states variable
    if "annual_states" in ds.data_vars:
        annual_states = ds["annual_states"]
        expected_dims = ["ensemble", "year"]
        if list(annual_states.dims) != expected_dims:
            results["errors"].append(f"Annual states dimensions {list(annual_states.dims)} don't match expected {expected_dims}")
            results["valid"] = False
        
        # Check valid_range attribute
        if "valid_range" in annual_states.attrs:
            valid_range = annual_states.attrs["valid_range"]
            # Convert to list for comparison to handle numpy arrays
            valid_range_list = list(valid_range) if hasattr(valid_range, '__iter__') else [valid_range]
            if valid_range_list != [0, 1]:
                results["warnings"].append(f"Annual states valid_range should be [0, 1], found: {valid_range}")
        else:
            results["warnings"].append("Annual states missing valid_range attribute")
        
        # Check other required attributes
        required_attrs = ["long_name", "description", "standard_name"]
        for attr in required_attrs:
            if attr not in annual_states.attrs:
                results["warnings"].append(f"Annual states missing {attr} attribute")
    
    # Validate hmm_parameters variable
    if "hmm_parameters" in ds.data_vars:
        hmm_parameters = ds["hmm_parameters"]
        expected_dims = ["ensemble", "parameter"]
        if list(hmm_parameters.dims) != expected_dims:
            results["errors"].append(f"HMM parameters dimensions {list(hmm_parameters.dims)} don't match expected {expected_dims}")
            results["valid"] = False
        
        # Check required attributes
        required_attrs = ["long_name", "description"]
        for attr in required_attrs:
            if attr not in hmm_parameters.attrs:
                results["warnings"].append(f"HMM parameters missing {attr} attribute")
    
    # Validate coordinate variables
    coord_checks = {
        "ensemble": "Ensemble member index",
        "time": "Monthly time steps", 
        "site": "Streamflow gage site names",
        "year": "Year labels for annual states",
        "parameter": "HMM parameter labels"
    }
    
    for coord, description in coord_checks.items():
        if coord in ds.coords:
            coord_var = ds[coord]
            if "long_name" not in coord_var.attrs:
                results["warnings"].append(f"Coordinate {coord} missing long_name attribute")
        else:
            results["warnings"].append(f"Coordinate variable {coord} not found")
    
    # Check right_id coordinate if shortage data exists
    if has_shortage and "right_id" in ds.coords:
        right_id = ds["right_id"]
        if "long_name" not in right_id.attrs:
            results["warnings"].append("Coordinate right_id missing long_name attribute")
    
    # Check for diversion variables (added by ii_process_diversions_reservoirs.py)
    diversion_vars = [var for var in ds.data_vars if var.startswith('diversion_')]
    if diversion_vars:
        print(f"    Found {len(diversion_vars)} diversion variables")
        for var in diversion_vars:
            diversion = ds[var]
            expected_dims = ["ensemble", "time", "right_id"]
            if list(diversion.dims) != expected_dims:
                results["errors"].append(f"Diversion {var} dimensions {list(diversion.dims)} don't match expected {expected_dims}")
                results["valid"] = False
            
            # Check required attributes
            required_attrs = ["long_name", "units", "description", "standard_name"]
            for attr in required_attrs:
                if attr not in diversion.attrs:
                    results["warnings"].append(f"Diversion {var} missing {attr} attribute")
    
    # Check for reservoir variables (added by ii_process_diversions_reservoirs.py)
    reservoir_vars = [var for var in ds.data_vars if var.startswith('reservoir_')]
    if reservoir_vars:
        print(f"    Found {len(reservoir_vars)} reservoir variables")
        for var in reservoir_vars:
            reservoir = ds[var]
            expected_dims = ["ensemble", "time", "reservoir_id"]
            if list(reservoir.dims) != expected_dims:
                results["errors"].append(f"Reservoir {var} dimensions {list(reservoir.dims)} don't match expected {expected_dims}")
                results["valid"] = False
            
            # Check required attributes
            required_attrs = ["long_name", "units", "description", "standard_name"]
            for attr in required_attrs:
                if attr not in reservoir.attrs:
                    results["warnings"].append(f"Reservoir {var} missing {attr} attribute")
    
    # Check for reservoir_id coordinate if reservoir data exists
    if reservoir_vars and "reservoir_id" in ds.coords:
        reservoir_id = ds["reservoir_id"]
        if "long_name" not in reservoir_id.attrs:
            results["warnings"].append("Coordinate reservoir_id missing long_name attribute")
    
    ds.close()
    
    return results


def print_summary(results: list):
    """Print a summary of validation results."""
    if not results:
        print("No validation results to summarize")
        return
    
    total_files = len(results)
    valid_files = sum(1 for r in results if r["valid"])
    invalid_files = total_files - valid_files
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total files validated: {total_files}")
    print(f"Valid files: {valid_files}")
    print(f"Invalid files: {invalid_files}")
    print(f"Success rate: {valid_files/total_files*100:.1f}%")
    
    if invalid_files > 0:
        print("\nINVALID FILES:")
        for result in results:
            if not result["valid"]:
                print(f"  - {result['file_path']}")
                for error in result["errors"]:
                    print(f"    ERROR: {error}")
    
    # Count warnings
    total_warnings = sum(len(r["warnings"]) for r in results)
    if total_warnings > 0:
        print(f"\nTotal warnings: {total_warnings}")
        print("\nFILES WITH WARNINGS:")
        for result in results:
            if result["warnings"]:
                print(f"  - {result['file_path']}")
                for warning in result["warnings"]:
                    print(f"    WARNING: {warning}")

### Main ###

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Validate WRAP outputs for specific filter-basin combinations')
    parser.add_argument('--filter', help='Filter name to process (e.g., basic, cooler, hotter)')
    parser.add_argument('--basin', help='Basin name to process (e.g., Colorado, Trinity, Brazos)')
    args = parser.parse_args()
    
    # Load basin configuration from JSON
    with open(basins_path, "r") as f:
        BASINS = json.load(f)

    # Load ensemble filters configuration from JSON
    with open(ensemble_filters_path, "r") as f:
        ENSEMBLE_CONFIG = json.load(f)

    results = []
    
    # Filter processing based on arguments
    if args.filter:
        filter_sets = [fs for fs in ENSEMBLE_CONFIG if fs["name"] == args.filter]
        if not filter_sets:
            print(f"Error: Filter '{args.filter}' not found in configuration")
            return 1
    else:
        filter_sets = ENSEMBLE_CONFIG
    
    if args.basin:
        if args.basin not in BASINS:
            print(f"Error: Basin '{args.basin}' not found in configuration")
            return 1
        basins = {args.basin: BASINS[args.basin]}
    else:
        basins = BASINS

    # Process selected combinations
    for filter_set in filter_sets:
        filter_name = filter_set["name"]
        print(f"Processing filter: {filter_name}")
        
        for basin_name, basin in basins.items():
            print(f"  Validating dataset for basin: {basin_name}")
            dataset_path = build_synthetic_dataset_path(filter_name, basin_name)
            if not dataset_path.exists():
                results.append({
                    "file_path": str(dataset_path),
                    "file_type": "synthetic_dataset",
                    "valid": False,
                    "errors": ["Synthetic streamflow NetCDF file not found"],
                    "warnings": []
                })
            else:
                result = validate_synthetic_dataset(dataset_path)
                results.append(result)
    
    # Print summary
    print_summary(results)
    
    # Return exit code based on validation results
    invalid_count = sum(1 for r in results if not r["valid"])
    return 1 if invalid_count > 0 else 0

if __name__ == "__main__":
    sys.exit(main())

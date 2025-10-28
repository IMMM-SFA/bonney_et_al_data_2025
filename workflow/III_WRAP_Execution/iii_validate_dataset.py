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
ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters.json"
hmm_metadata_path = repo_data_path / "configs" / "hmm_synthetic_data_metadata.json"
wrap_metadata_path = repo_data_path / "configs" / "wrap_variable_metadata.json"

### Functions ###

def load_metadata():
    """Load metadata from JSON files."""
    with open(hmm_metadata_path, 'r') as f:
        hmm_metadata = json.load(f)
    
    with open(wrap_metadata_path, 'r') as f:
        wrap_metadata = json.load(f)
    
    return hmm_metadata, wrap_metadata

def get_expected_variables(hmm_metadata, wrap_metadata):
    """Extract expected variable names from metadata."""
    expected_vars = {}
    
    # HMM variables (exclude coordinate_variables)
    for key, value in hmm_metadata.items():
        if key != "coordinate_variables" and isinstance(value, dict) and "long_name" in value:
            expected_vars[key] = value
    
    # WRAP diversion variables
    if "diversion" in wrap_metadata:
        for var_name, metadata in wrap_metadata["diversion"].items():
            expected_vars[var_name] = {"metadata": metadata, "type": "diversion"}
    
    # WRAP reservoir variables
    if "reservoir" in wrap_metadata:
        for var_name, metadata in wrap_metadata["reservoir"].items():
            expected_vars[var_name] = {"metadata": metadata, "type": "reservoir"}
    
    return expected_vars

def get_expected_coordinates(hmm_metadata, wrap_metadata):
    """Extract expected coordinate variable names from metadata."""
    expected_coords = {}
    
    # Get coordinates from HMM metadata
    if "coordinate_variables" in hmm_metadata:
        expected_coords.update(hmm_metadata["coordinate_variables"])
    
    # Get additional coordinates from WRAP metadata
    if "coordinate_variables" in wrap_metadata:
        for coord_name, metadata in wrap_metadata["coordinate_variables"].items():
            if coord_name not in expected_coords:
                expected_coords[coord_name] = metadata
    
    return expected_coords

def build_synthetic_dataset_path(filter_name: str, basin_name: str) -> Path:
    return outputs_path / "bayesian_hmm" / f"{filter_name}" / f"{basin_name.lower()}" / f"{filter_name}_{basin_name.lower()}_synthetic_dataset.nc"

def validate_synthetic_dataset(file_path: Path, hmm_metadata: dict, wrap_metadata: dict) -> dict:
    """Validate synthetic streamflow dataset structure using metadata files."""
    print(f"  Validating synthetic streamflow: {file_path.name}")
    
    ds = xr.open_dataset(file_path)
    results = {
        "file_path": str(file_path),
        "file_type": "synthetic_dataset",
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Get expected variables and coordinates from metadata
    expected_vars = get_expected_variables(hmm_metadata, wrap_metadata)
    expected_coords = get_expected_coordinates(hmm_metadata, wrap_metadata)
    
    # Check required coordinate variables (dimensions)
    for coord_name, coord_metadata in expected_coords.items():
        if coord_name not in ds.dims and coord_name not in ds.coords:
            # Some coordinates might be optional depending on data presence
            if coord_name in ["right_id", "reservoir_id"]:
                # Check if there are any diversion or reservoir variables in the dataset
                has_diversion_data = any(isinstance(v, dict) and v.get("type") == "diversion" for v in expected_vars.values() 
                                        if v in [expected_vars.get(var) for var in ds.data_vars])
                has_reservoir_data = any(isinstance(v, dict) and v.get("type") == "reservoir" for v in expected_vars.values()
                                        if v in [expected_vars.get(var) for var in ds.data_vars])
                
                if coord_name == "right_id" and has_diversion_data:
                    results["errors"].append(f"Missing required coordinate: {coord_name} (needed for diversion data)")
                    results["valid"] = False
                elif coord_name == "reservoir_id" and has_reservoir_data:
                    results["errors"].append(f"Missing required coordinate: {coord_name} (needed for reservoir data)")
                    results["valid"] = False
            elif coord_name == "hmm_parameter_name":
                # Check if it exists under different name (parameter)
                if "parameter" not in ds.dims and "parameter" not in ds.coords:
                    results["warnings"].append(f"Coordinate {coord_name} (or 'parameter') not found")
            else:
                # Core coordinates should always be present
                results["warnings"].append(f"Missing coordinate: {coord_name}")
    
    # Check required data variables from metadata
    for var_name, var_info in expected_vars.items():
        # Handle WRAP variables (which have metadata and type)
        if isinstance(var_info, dict) and "type" in var_info:
            var_type = var_info["type"]
            # WRAP variables - check if they exist
            if var_name not in ds.data_vars:
                results["errors"].append(f"Missing required {var_type} variable: {var_name}")
                results["valid"] = False
        else:
            # HMM variables - these are always required
            if var_name not in ds.data_vars:
                # Handle alternate naming (e.g., streamflow vs synthetic_streamflow)
                alt_names = {
                    "synthetic_streamflow": "streamflow",
                    "annual_wet_dry_state": "annual_states"
                }
                alt_name = alt_names.get(var_name, None)
                if alt_name and alt_name in ds.data_vars:
                    results["warnings"].append(f"Variable {var_name} found as {alt_name}")
                else:
                    results["errors"].append(f"Missing required data variable: {var_name}")
                    results["valid"] = False
    
    # Validate data variable attributes using metadata
    for var_name in ds.data_vars:
        # Check if this variable should have metadata
        var_metadata = None
        
        # Check if it's in expected vars
        if var_name in expected_vars:
            var_info = expected_vars[var_name]
            # Extract metadata from WRAP variables (which have nested structure)
            if isinstance(var_info, dict) and "metadata" in var_info:
                var_metadata = var_info["metadata"]
            else:
                var_metadata = var_info
        # Check alternate names for HMM variables
        elif var_name == "streamflow" and "synthetic_streamflow" in expected_vars:
            var_metadata = expected_vars["synthetic_streamflow"]
        elif var_name == "annual_states" and "annual_wet_dry_state" in expected_vars:
            var_metadata = expected_vars["annual_wet_dry_state"]
        
        if var_metadata:
            var_obj = ds[var_name]
            
            # Check units attribute
            if "units" in var_metadata:
                expected_units = var_metadata["units"]
                if "units" in var_obj.attrs:
                    actual_units = var_obj.attrs["units"]
                    # Simple check - just warn if they don't contain similar keywords
                    if expected_units not in actual_units and actual_units not in expected_units:
                        results["warnings"].append(f"Variable {var_name} units may not match: expected '{expected_units}', found '{actual_units}'")
                else:
                    results["warnings"].append(f"Variable {var_name} missing units attribute")
            
            # Check long_name attribute
            if "long_name" not in var_obj.attrs:
                results["warnings"].append(f"Variable {var_name} missing long_name attribute")
            
            # Check description attribute
            if "description" not in var_obj.attrs:
                results["warnings"].append(f"Variable {var_name} missing description attribute")
    
    # Validate coordinate variable attributes using metadata
    for coord_name in ds.coords:
        if coord_name in expected_coords:
            coord_metadata = expected_coords[coord_name]
            coord_obj = ds[coord_name]
            
            # Check long_name attribute
            if "long_name" not in coord_obj.attrs:
                results["warnings"].append(f"Coordinate {coord_name} missing long_name attribute")
        # Handle alternate coordinate names
        elif coord_name in ["ensemble", "time", "site", "parameter"]:
            # These are common coordinates that might not be in metadata yet
            if "long_name" not in ds[coord_name].attrs:
                results["warnings"].append(f"Coordinate {coord_name} missing long_name attribute")
    
    # Count variable types for reporting
    diversion_vars = [var for var, info in expected_vars.items() 
                     if isinstance(info, dict) and info.get("type") == "diversion" and var in ds.data_vars]
    reservoir_vars = [var for var, info in expected_vars.items() 
                     if isinstance(info, dict) and info.get("type") == "reservoir" and var in ds.data_vars]
    
    if diversion_vars:
        print(f"    Found {len(diversion_vars)} diversion variables")
    
    if reservoir_vars:
        print(f"    Found {len(reservoir_vars)} reservoir variables")
    
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
    
    # Load metadata files
    print("Loading metadata files...")
    hmm_metadata, wrap_metadata = load_metadata()
    print(f"  Loaded {len([k for k in hmm_metadata.keys() if k != 'coordinate_variables'])} HMM variables")
    print(f"  Loaded {len(wrap_metadata.get('diversion', {}))} diversion variables")
    print(f"  Loaded {len(wrap_metadata.get('reservoir', {}))} reservoir variables")

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
                result = validate_synthetic_dataset(dataset_path, hmm_metadata, wrap_metadata)
                results.append(result)
    
    # Print summary
    print_summary(results)
    
    # Return exit code based on validation results
    invalid_count = sum(1 for r in results if not r["valid"])
    return 1 if invalid_count > 0 else 0

if __name__ == "__main__":
    sys.exit(main())

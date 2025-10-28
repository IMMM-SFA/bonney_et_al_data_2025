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
    print(f"\n{'='*80}")
    print(f"Validating: {file_path.name}")
    print(f"{'='*80}")
    
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
    
    print(f"\nChecking {len(expected_vars)} expected variables...")
    print(f"-" * 80)
    
    # Check each expected variable
    for var_name, var_info in expected_vars.items():
        # Extract metadata and type
        if isinstance(var_info, dict) and "metadata" in var_info:
            var_metadata = var_info["metadata"]
            var_type = var_info.get("type", "unknown")
        else:
            var_metadata = var_info
            var_type = "hmm"
        
        # Check if variable exists (handle alternate names)
        actual_var_name = var_name
        if var_name not in ds.data_vars:
            # Try alternate names for HMM variables
            alt_names = {
                "synthetic_streamflow": "streamflow",
                "annual_wet_dry_state": "annual_states"
            }
            if var_name in alt_names and alt_names[var_name] in ds.data_vars:
                actual_var_name = alt_names[var_name]
                results["warnings"].append(f"Variable '{var_name}' found as '{actual_var_name}'")
            else:
                results["errors"].append(f"Missing required {var_type} variable: {var_name}")
                results["valid"] = False
                print(f"\n❌ {var_name} ({var_type}): MISSING")
                continue
        
        # Variable exists - log information
        var_obj = ds[actual_var_name]
        print(f"\n✓ {actual_var_name} ({var_type})")
        print(f"  Dimensions: {list(var_obj.dims)} {var_obj.shape}")
        
        # Get sample values and statistics
        try:
            values = var_obj.values.flatten()
            valid_values = values[~np.isnan(values)]
            
            if len(valid_values) > 0:
                sample_size = min(10, len(valid_values))
                sample_indices = np.linspace(0, len(valid_values)-1, sample_size, dtype=int)
                samples = valid_values[sample_indices]
                mean_val = np.nanmean(values)
                
                print(f"  Sample values (n={sample_size}): {samples}")
                print(f"  Mean: {mean_val:.6f}")
            else:
                print(f"  Warning: All values are NaN")
                results["warnings"].append(f"Variable '{actual_var_name}' contains only NaN values")
        except Exception as e:
            print(f"  Warning: Could not compute statistics: {e}")
        
        # Check metadata attributes
        missing_attrs = []
        if "long_name" not in var_obj.attrs:
            missing_attrs.append("long_name")
        if "units" not in var_obj.attrs:
            missing_attrs.append("units")
        if "description" not in var_obj.attrs:
            missing_attrs.append("description")
        
        if missing_attrs:
            results["warnings"].append(f"Variable '{actual_var_name}' missing attributes: {', '.join(missing_attrs)}")
            print(f"  Missing attributes: {', '.join(missing_attrs)}")
    
    # Check coordinates
    print(f"\n{'-' * 80}")
    print(f"Checking {len(expected_coords)} expected coordinates...")
    print(f"-" * 80)
    
    for coord_name, coord_metadata in expected_coords.items():
        # Handle alternate coordinate names
        actual_coord_name = coord_name
        if coord_name not in ds.coords and coord_name not in ds.dims:
            # Check alternate names
            if coord_name == "hmm_parameter_name" and "parameter" in ds.coords:
                actual_coord_name = "parameter"
            elif coord_name in ["right_id", "reservoir_id"]:
                # These are optional depending on data presence
                continue
            else:
                results["warnings"].append(f"Coordinate '{coord_name}' not found")
                print(f"\n⚠ {coord_name}: NOT FOUND")
                continue
        
        # Coordinate exists
        if actual_coord_name in ds.coords:
            coord_obj = ds[actual_coord_name]
            print(f"\n✓ {actual_coord_name}")
            print(f"  Size: {coord_obj.size}")
            
            # Check for long_name attribute
            if "long_name" not in coord_obj.attrs:
                results["warnings"].append(f"Coordinate '{actual_coord_name}' missing long_name attribute")
                print(f"  Missing attribute: long_name")
    
    ds.close()
    
    return results


def print_summary(results: list):
    """Print a summary of validation results."""
    if not results:
        print("\nNo validation results to summarize")
        return
    
    total_files = len(results)
    valid_files = sum(1 for r in results if r["valid"])
    invalid_files = total_files - valid_files
    total_errors = sum(len(r["errors"]) for r in results)
    total_warnings = sum(len(r["warnings"]) for r in results)
    
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Total files validated: {total_files}")
    print(f"Valid files: {valid_files}")
    print(f"Invalid files: {invalid_files}")
    print(f"Total errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")
    print(f"Success rate: {valid_files/total_files*100:.1f}%")
    
    if total_errors > 0:
        print("\n" + "-"*80)
        print("ERRORS:")
        print("-"*80)
        for result in results:
            if result["errors"]:
                print(f"\n{result['file_path']}:")
                for error in result["errors"]:
                    print(f"  ❌ {error}")
    
    if total_warnings > 0:
        print("\n" + "-"*80)
        print("WARNINGS:")
        print("-"*80)
        for result in results:
            if result["warnings"]:
                print(f"\n{result['file_path']}:")
                for warning in result["warnings"]:
                    print(f"  ⚠ {warning}")

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

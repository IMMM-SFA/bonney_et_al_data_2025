"""
Dataset Validation Script for Workflow Outputs

This script validates that the produced .nc datasets contain the expected fields
and structure according to the workflow documentation.

Validates:
1. 9505 Master Streamflow datasets (CFS and acre-feet versions)
2. HMM Model datasets
3. Synthetic Streamflow datasets (when available)
"""

import xarray as xr
import numpy as np
import pandas as pd
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

def validate_9505_master_streamflow(file_path: Path) -> dict:
    """Validate 9505 master streamflow dataset structure."""
    print(f"  Validating 9505 master streamflow: {file_path.name}")
    
    try:
        ds = xr.open_dataset(file_path)
        results = {
            "file_path": str(file_path),
            "file_type": "9505_master_streamflow",
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required dimensions
        required_dims = ["ensemble_id", "time_mn"]
        for dim in required_dims:
            if dim not in ds.dims:
                results["errors"].append(f"Missing required dimension: {dim}")
                results["valid"] = False
        
        # Check for reach variables
        reach_vars = [var for var in ds.data_vars if var.startswith('reach_')]
        if not reach_vars:
            results["errors"].append("No reach variables found")
            results["valid"] = False
        else:
            print(f"    Found {len(reach_vars)} reach variables")
        
        # Check units for acre-feet version
        if "_af.nc" in str(file_path):
            sample_reach = reach_vars[0] if reach_vars else None
            if sample_reach and "units" in ds[sample_reach].attrs:
                units = ds[sample_reach].attrs["units"]
                if "acre-feet" not in units.lower():
                    results["warnings"].append(f"Expected acre-feet units, found: {units}")
        
        ds.close()
        
    except Exception as e:
        results = {
            "file_path": str(file_path),
            "file_type": "9505_master_streamflow",
            "valid": False,
            "errors": [f"Failed to open dataset: {str(e)}"],
            "warnings": []
        }
    
    return results

def validate_hmm_model(file_path: Path) -> dict:
    """Validate HMM model dataset structure."""
    print(f"  Validating HMM model: {file_path.name}")
    
    try:
        ds = xr.open_dataset(file_path)
        results = {
            "file_path": str(file_path),
            "file_type": "hmm_model",
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # HMM models can have various structures
        if len(ds.data_vars) == 0:
            results["warnings"].append("No data variables found in HMM model")
        
        # Check for common HMM model variables
        expected_vars = ["mu", "sigma", "transition_mat", "initial_dist"]
        found_vars = [var for var in expected_vars if var in ds.data_vars]
        if found_vars:
            print(f"    Found HMM parameters: {found_vars}")
        
        ds.close()
        
    except Exception as e:
        results = {
            "file_path": str(file_path),
            "file_type": "hmm_model",
            "valid": False,
            "errors": [f"Failed to open dataset: {str(e)}"],
            "warnings": []
        }
    
    return results

def validate_synthetic_streamflow(file_path: Path) -> dict:
    """Validate synthetic streamflow dataset structure."""
    print(f"  Validating synthetic streamflow: {file_path.name}")
    
    try:
        ds = xr.open_dataset(file_path)
        results = {
            "file_path": str(file_path),
            "file_type": "synthetic_streamflow",
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required dimensions
        required_dims = ["ensemble", "time", "site"]
        for dim in required_dims:
            if dim not in ds.dims:
                results["errors"].append(f"Missing required dimension: {dim}")
                results["valid"] = False
        
        # Check required data variables
        required_data_vars = ["streamflow", "annual_states", "hmm_parameters"]
        for var in required_data_vars:
            if var not in ds.data_vars:
                results["errors"].append(f"Missing required data variable: {var}")
                results["valid"] = False
        
        # Check optional data variables
        if "shortage" not in ds.data_vars:
            results["warnings"].append("Optional data variable not found: shortage")
        
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
        
        # Validate annual_states variable
        if "annual_states" in ds.data_vars:
            annual_states = ds["annual_states"]
            expected_dims = ["ensemble", "year"]
            if list(annual_states.dims) != expected_dims:
                results["errors"].append(f"Annual states dimensions {list(annual_states.dims)} don't match expected {expected_dims}")
                results["valid"] = False
        
        ds.close()
        
    except Exception as e:
        results = {
            "file_path": str(file_path),
            "file_type": "synthetic_streamflow",
            "valid": False,
            "errors": [f"Failed to open dataset: {str(e)}"],
            "warnings": []
        }
    
    return results

def validate_dataset(file_path: Path) -> dict:
    """Determine file type and validate accordingly."""
    if not file_path.exists():
        return {
            "file_path": str(file_path),
            "valid": False,
            "errors": ["File does not exist"],
            "warnings": []
        }
    
    # Determine file type based on path and name
    if "master_streamflow" in file_path.name:
        return validate_9505_master_streamflow(file_path)
    elif "model.nc" in file_path.name:
        return validate_hmm_model(file_path)
    elif "synthetic_streamflow" in file_path.name:
        return validate_synthetic_streamflow(file_path)
    else:
        # Try to determine type by examining the file
        try:
            ds = xr.open_dataset(file_path)
            data_vars = list(ds.data_vars)
            coords = list(ds.coords)
            ds.close()
            
            # Heuristic classification
            if "streamflow" in data_vars and "ensemble" in coords:
                return validate_synthetic_streamflow(file_path)
            elif any(var.startswith("reach_") for var in data_vars):
                return validate_9505_master_streamflow(file_path)
            else:
                return validate_hmm_model(file_path)
        except:
            return {
                "file_path": str(file_path),
                "valid": False,
                "errors": ["Could not determine file type or open file"],
                "warnings": []
            }

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
    parser = argparse.ArgumentParser(description='Validate .nc datasets for specific filter-basin combinations')
    parser.add_argument('--filter', help='Filter name to process (e.g., basic, cooler, hotter)')
    parser.add_argument('--basin', help='Basin name to process (e.g., Colorado, Trinity, Brazos)')
    parser.add_argument('--all', action='store_true', help='Validate all datasets in outputs directory')
    args = parser.parse_args()
    
    # Load basin configuration from JSON
    with open(basins_path, "r") as f:
        BASINS = json.load(f)

    # Load ensemble filters configuration from JSON
    with open(ensemble_filters_path, "r") as f:
        ENSEMBLE_CONFIG = json.load(f)

    results = []
    
    if args.all:
        # Validate all datasets in outputs directory
        print("Validating all datasets in outputs directory...")
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            nc_files = list(outputs_dir.rglob("*.nc"))
            print(f"Found {len(nc_files)} .nc files")
            
            for file_path in nc_files:
                result = validate_dataset(file_path)
                results.append(result)
        else:
            print("Outputs directory not found")
            return 1
    else:
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
                print(f"  Validating datasets for basin: {basin_name}")
                
                # Validate 9505 master streamflow datasets
                master_streamflow_dir = outputs_path / "9505" / "reach_subset_combined"
                if master_streamflow_dir.exists():
                    for nc_file in master_streamflow_dir.glob("*.nc"):
                        result = validate_dataset(nc_file)
                        results.append(result)
                
                # Validate HMM model datasets
                hmm_model_path = outputs_path / "bayesian_hmm" / f"{filter_name}" / f"{basin_name.lower()}" / f"{basin_name}_{filter_name}_model.nc"
                if hmm_model_path.exists():
                    result = validate_dataset(hmm_model_path)
                    results.append(result)
                
                # Validate synthetic streamflow datasets
                synthetic_streamflow_path = outputs_path / "bayesian_hmm" / f"{filter_name}" / f"{basin_name.lower()}" / f"{filter_name}_{basin_name.lower()}_synthetic_streamflow.nc"
                if synthetic_streamflow_path.exists():
                    result = validate_dataset(synthetic_streamflow_path)
                    results.append(result)
    
    # Print summary
    print_summary(results)
    
    # Return exit code based on validation results
    invalid_count = sum(1 for r in results if not r["valid"])
    return 1 if invalid_count > 0 else 0

if __name__ == "__main__":
    sys.exit(main())

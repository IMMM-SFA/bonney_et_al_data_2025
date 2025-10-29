"""
This script optimizes NetCDF files by:
1. Converting data types to smaller precision where appropriate
2. Adding internal compression using zlib with shuffle filter
3. Reporting size reduction statistics

Best Practices:
- float64 → float32 (sufficient for most scientific data)
- Internal compression with shuffle filter (transparent to users)
- Compression level 4-5 (good balance of size/speed)
"""

import os
import multiprocessing
import numpy as np
import xarray as xr
from pathlib import Path
import json
import argparse
import shutil

from toolkit import repo_data_path, outputs_path

### Settings ###
# Compression settings
COMPRESSION_LEVEL = 5  # 1-9, where 5 is a good balance
USE_SHUFFLE = True     # Reorganize bytes for better compression

# Use a conservative number of processes to avoid system freeze
# NetCDF operations can be I/O intensive
num_processes = 4

# Archive path (set to None to skip archive creation)
archive_path = outputs_path / "data_archive"

### Path Configuration ###
basins_path = repo_data_path / "configs" / "basins.json"
ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters.json"
readme_source = Path(__file__).parent.parent / "MSD-README.md"

### Functions ###

def copy_directory(src, dst):
    """Copy directory recursively."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def get_file_size_mb(path):
    """Get file size in MB."""
    return os.path.getsize(path) / (1024 * 1024)

def optimize_dtype(da, var_name):
    """
    Optimize data type for a data array.
    
    Parameters
    ----------
    da : xr.DataArray
        Input data array
    var_name : str
        Variable name for reporting
    
    Returns
    -------
    tuple
        (optimized_array, converted, original_dtype)
    """
    original_dtype = da.dtype
    
    # Skip if already optimal or is a string
    if da.dtype == np.float32 or da.dtype == np.int32:
        return da, False, original_dtype
    
    if da.dtype.kind == 'U' or da.dtype.kind == 'S':
        return da, False, original_dtype
    
    # Convert floating point to float32
    if da.dtype.kind == 'f':
        converted = da.astype(np.float32)
        
        # Check for overflow/underflow
        original_finite = np.isfinite(da.values)
        converted_finite = np.isfinite(converted.values)
        if not np.array_equal(original_finite, converted_finite):
            print(f"    WARNING: {var_name} - float32 conversion created inf/nan values")
            return da, False, original_dtype
        
        return converted, True, original_dtype
    
    # Convert integers to int32 if they fit
    if da.dtype.kind == 'i':
        if da.dtype == np.int64:
            da_min = float(da.min().values)
            da_max = float(da.max().values)
            
            if da_min >= np.iinfo(np.int32).min and da_max <= np.iinfo(np.int32).max:
                return da.astype(np.int32), True, original_dtype
    
    return da, False, original_dtype

def optimize_single_file(input_path, output_path):
    """
    Optimize a single NetCDF file with dtype conversion and compression.
    
    Parameters
    ----------
    input_path : Path
        Input NetCDF file path
    output_path : Path
        Output NetCDF file path
    
    Returns
    -------
    dict
        Results dictionary with statistics
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    print(f"\nOptimizing: {input_path.name}")
    print("-" * 80)
    
    # Get original size
    original_size = get_file_size_mb(input_path)
    print(f"Original size: {original_size:.2f} MB")
    
    # Load dataset
    ds = xr.open_dataset(input_path)
    
    # Track conversions
    conversions = []
    
    # Optimize data variables
    print("\nOptimizing data variables:")
    for var_name in ds.data_vars:
        da = ds[var_name]
        optimized_da, converted, original_dtype = optimize_dtype(da, var_name)
        
        if converted:
            print(f"  {var_name}: {original_dtype} → {optimized_da.dtype}")
            conversions.append((var_name, original_dtype, optimized_da.dtype))
            ds[var_name] = optimized_da
        else:
            print(f"  {var_name}: {da.dtype} (no change)")
    
    # Optimize coordinates
    print("\nOptimizing coordinates:")
    for coord_name in ds.coords:
        da = ds[coord_name]
        optimized_da, converted, original_dtype = optimize_dtype(da, coord_name)
        
        if converted:
            print(f"  {coord_name}: {original_dtype} → {optimized_da.dtype}")
            conversions.append((coord_name, original_dtype, optimized_da.dtype))
            ds[coord_name] = optimized_da
        else:
            print(f"  {coord_name}: {da.dtype} (no change)")
    
    # Set up encoding with compression
    print("\nApplying compression settings:")
    print(f"  Compression level: {COMPRESSION_LEVEL}")
    print(f"  Shuffle filter: {USE_SHUFFLE}")
    print(f"  Chunking: Along realization axis")
    
    encoding = {}
    for var_name in list(ds.data_vars) + list(ds.coords):
        var_obj = ds[var_name]
        
        # Chunking along realization axis
        chunks = None
        if var_name in ds.data_vars and len(var_obj.dims) > 0:
            # For variables with realization dimension, chunk along it
            if 'realization' in var_obj.dims:
                chunk_list = []
                for dim in var_obj.dims:
                    if dim == 'realization':
                        # Chunk 1 realization at a time
                        chunk_list.append(1)
                    else:
                        # Keep other dimensions unchunked (full size)
                        chunk_list.append(var_obj.shape[var_obj.dims.index(dim)])
                chunks = tuple(chunk_list)
                print(f"  {var_name}: chunks={chunks}")
        
        encoding[var_name] = {
            'zlib': True,
            'complevel': COMPRESSION_LEVEL,
            'shuffle': USE_SHUFFLE
        }
        
        if chunks is not None:
            encoding[var_name]['chunksizes'] = chunks
    
    # Save optimized file
    print(f"\nSaving optimized file to: {output_path.name}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        ds.to_netcdf(output_path, encoding=encoding, format='NETCDF4')
        ds.close()
        
        # Verify the output file was created and is readable
        print(f"Verifying optimized file...")
        test_ds = xr.open_dataset(output_path)
        test_ds.close()
        
        # Report results
        optimized_size = get_file_size_mb(output_path)
        reduction = original_size - optimized_size
        reduction_pct = (reduction / original_size) * 100
        
        print(f"\nOptimization complete!")
        print(f"  Original size:  {original_size:.2f} MB")
        print(f"  Optimized size: {optimized_size:.2f} MB")
        print(f"  Reduction:      {reduction:.2f} MB ({reduction_pct:.1f}%)")
        
        return {
            'file': str(output_path),
            'success': True,
            'original_size': original_size,
            'optimized_size': optimized_size,
            'reduction': reduction,
            'reduction_pct': reduction_pct,
            'conversions': len(conversions)
        }
        
    except Exception as e:
        print(f"\n❌ ERROR during optimization: {e}")
        
        return {
            'file': str(output_path),
            'success': False,
            'error': str(e)
        }

def process_filter_basin_combination(args):
    """
    Worker function to optimize a single filter-basin combination.
    
    Parameters
    ----------
    args : tuple
        Contains (filter_name, basin_name)
    
    Returns
    -------
    dict
        Results dictionary
    """
    filter_name, basin_name = args
    
    # Construct input NetCDF file path
    nc_filename = f"{filter_name}_{basin_name.lower()}_synthetic_dataset.nc"
    input_path = outputs_path / "bayesian_hmm" / filter_name / basin_name.lower() / nc_filename
    
    if not input_path.exists():
        print(f"\nSkipping (not found): {nc_filename}")
        return {
            'file': str(input_path),
            'success': False,
            'error': 'File not found'
        }
    
    # Construct output path in archive destination
    output_path = archive_path / basin_name / nc_filename
    
    return optimize_single_file(input_path, output_path)

### Main ###

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optimize NetCDF files and prepare data archive')
    parser.add_argument('--filter', help='Filter name to process (e.g., All Models, Bias Correction - Daymet)')
    parser.add_argument('--basin', help='Basin name to process (e.g., Colorado, Trinity, Sabine)')
    args = parser.parse_args()
    
    # Load basin configuration
    with open(basins_path, "r") as f:
        BASINS = json.load(f)
    
    # Load ensemble filters configuration
    with open(ensemble_filters_path, "r") as f:
        ENSEMBLE_CONFIG = json.load(f)
    
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
    
    # Collect all filter-basin combinations
    all_combinations = []
    for filter_set in filter_sets:
        filter_name = filter_set["name"]
        for basin_name in basins.keys():
            combination = (filter_name, basin_name)
            all_combinations.append(combination)
    
    print("="*80)
    print("OPTIMIZING NETCDF DATASETS")
    print("="*80)
    print(f"Processing {len(all_combinations)} filter-basin combinations...")
    print(f"Using {num_processes} parallel processes")
    print("="*80)
    
    # Process all combinations in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_filter_basin_combination, all_combinations)
    
    # Print summary
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Total files: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        total_original = sum(r['original_size'] for r in successful)
        total_optimized = sum(r['optimized_size'] for r in successful)
        total_reduction = total_original - total_optimized
        total_reduction_pct = (total_reduction / total_original) * 100 if total_original > 0 else 0
        
        print(f"\nTotal original size:  {total_original:.2f} MB")
        print(f"Total optimized size: {total_optimized:.2f} MB")
        print(f"Total reduction:      {total_reduction:.2f} MB ({total_reduction_pct:.1f}%)")
    
    if failed:
        print(f"\nFailed files:")
        for r in failed:
            print(f"  ❌ {Path(r['file']).name}: {r.get('error', 'Unknown error')}")
        raise Exception("Failed to optimize some files")
    
    # Finalize archive with README and data folder
    print("\n" + "="*80)
    print("FINALIZING DATA ARCHIVE")
    print("="*80)
    
    output_dir = Path(archive_path)
    
    # Copy data folder
    print("\nCopying data folder...")
    data_dest = output_dir / "data"
    copy_directory(repo_data_path, data_dest)
    print(f"  Copied: data/ -> {data_dest.name}/")
    
    # Copy README
    print("\nCopying README...")
    readme_dest = output_dir / "README.md"
    shutil.copy(readme_source, readme_dest)
    print(f"  Copied: {readme_source.name} -> {readme_dest.name}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("ARCHIVE COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"\nStructure:")
    print(f"  {output_dir.name}/")
    print(f"    README.md")
    print(f"    data/")
    for basin_name in basins.keys():
        basin_folder = output_dir / basin_name
        if basin_folder.exists():
            nc_count = len(list(basin_folder.glob("*.nc")))
            print(f"    {basin_name}/ ({nc_count} datasets)")
    
    return 0 if len(failed) == 0 else 1

if __name__ == "__main__":
    exit(main())

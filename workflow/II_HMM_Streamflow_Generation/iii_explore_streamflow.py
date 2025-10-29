"""
This script loads synthetic streamflow data and produces exploratory plots.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse

from toolkit.data.io import load_netcdf_format
from toolkit.wrap.io import flo_to_df
from toolkit import repo_data_path, outputs_path

sns.set_style("whitegrid")

basins_path = repo_data_path / "configs" / "basins.json"
ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters.json"
output_dir = outputs_path / "bayesian_hmm"

def load_synthetic_data(basin_name, filter_name):
    """Load synthetic streamflow data from NetCDF file."""
    synthetic_nc_path = output_dir / f"{filter_name}" / f"{basin_name.lower()}" / f"{filter_name}_{basin_name.lower()}_synthetic_dataset.nc"
    
    if not synthetic_nc_path.exists():
        print(f"Synthetic data not found at {synthetic_nc_path}")
        return None
    
    data_dict = load_netcdf_format(str(synthetic_nc_path))
    return data_dict

def load_historical_data(basin):
    """Load historical streamflow data from FLO file."""
    flo_file = repo_data_path / basin["flo_file"]
    hist_monthly = flo_to_df(str(flo_file))
    return hist_monthly

def plot_average_annual_streamflow(synthetic_data, basin_name, filter_name, output_path):
    """Plot annual streamflow for each gage across years using first realization."""
    streamflow = synthetic_data['streamflow']  # (n_realizations, n_months, n_sites)
    time_index = pd.to_datetime(synthetic_data['streamflow_index'])
    site_names = synthetic_data['streamflow_columns']
    
    # Get first realization
    first_realization = streamflow[0, :, :]  # (n_months, n_sites)
    n_months, n_sites = first_realization.shape
    n_years = n_months // 12
    
    # Reshape to (n_years, 12, n_sites) and sum over months
    streamflow_reshaped = first_realization.reshape(n_years, 12, n_sites)
    annual_streamflow = streamflow_reshaped.sum(axis=1)  # (n_years, n_sites)
    
    # Get years from time index
    years = np.arange(int(time_index[0].year), int(time_index[0].year) + n_years)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, site in enumerate(site_names):
        ax.plot(years, annual_streamflow[:, i], marker='o', linewidth=2)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Annual Streamflow (acre-feet)', fontsize=12)
    ax.set_title(f'Annual Streamflow - {basin_name} ({filter_name}) - First Realization', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved annual streamflow plot to {output_path}")

def plot_outflow_comparison(synthetic_data, hist_monthly, gage_name, basin_name, filter_name, output_path):
    """Plot historical vs synthetic annual streamflow at the outflow gage."""
    streamflow = synthetic_data['streamflow']  # (n_realizations, n_months, n_sites)
    time_index = pd.to_datetime(synthetic_data['streamflow_index'])
    site_names = synthetic_data['streamflow_columns']
    
    # Find the gage index
    gage_index = list(site_names).index(gage_name)
    
    # Get first realization for the outflow gage
    first_realization = streamflow[0, :, gage_index]  # (n_months,)
    
    # Calculate annual streamflow for synthetic
    n_months = len(first_realization)
    n_years = n_months // 12
    annual_synthetic = first_realization.reshape(n_years, 12).sum(axis=1)
    synthetic_years = np.arange(int(time_index[0].year), int(time_index[0].year) + n_years)
    
    # Calculate annual streamflow for historical
    hist_annual = hist_monthly[gage_name].resample('Y').sum()
    hist_years = hist_annual.index.year
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(hist_years, hist_annual.values, marker='o', linewidth=2, label='Historical', alpha=0.7)
    ax.plot(synthetic_years, annual_synthetic, marker='s', linewidth=2, label='Synthetic (First Realization)', alpha=0.7)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Annual Streamflow (acre-feet)', fontsize=12)
    ax.set_title(f'Outflow Gage Annual Streamflow Comparison - {basin_name} ({filter_name})\nGage: {gage_name}', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved outflow comparison plot to {output_path}")

def plot_correlation_matrix(streamflow, title, output_path, data_type='monthly'):
    """Create correlation matrix heatmap."""
    correlation_matrix = np.corrcoef(streamflow.T)
    
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cmap = plt.get_cmap("viridis")
    im = ax.matshow(correlation_matrix, cmap=cmap)
    fig.colorbar(im)
    
    ax.tick_params(axis="both", labelsize=14)
    ax.set_ylabel("Basin Node", fontsize=16)
    ax.set_xlabel("Basin Node", fontsize=16)
    ax.set_title(f'{title}\nCorrelation coefficient by gage site on {data_type} streamflow', 
                fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation plot to {output_path}")

def explore_streamflow(basin_name, basin, filter_name):
    """Generate all exploratory plots for a basin."""
    print(f"\nExploring streamflow for {basin_name} ({filter_name})")
    
    # Create output directory for plots
    plot_dir = output_dir / f"{filter_name}" / f"{basin_name.lower()}" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Load synthetic data
    synthetic_data = load_synthetic_data(basin_name, filter_name)
    if synthetic_data is None:
        return
    
    # Load historical data
    hist_monthly = load_historical_data(basin)
    
    # Get outflow gage name
    gage_name = basin["gage_name"]
    
    # 1. Plot average annual streamflow
    avg_annual_path = plot_dir / f"{filter_name}_{basin_name.lower()}_avg_annual_streamflow.png"
    plot_average_annual_streamflow(synthetic_data, basin_name, filter_name, avg_annual_path)
    
    # 2. Plot outflow gage comparison (historical vs synthetic)
    outflow_comparison_path = plot_dir / f"{filter_name}_{basin_name.lower()}_outflow_comparison.png"
    plot_outflow_comparison(synthetic_data, hist_monthly, gage_name, basin_name, filter_name, outflow_comparison_path)
    
    # 3. Correlation plots for synthetic data (first realization)
    streamflow = synthetic_data['streamflow']  # (n_realizations, n_months, n_sites)
    time_index = synthetic_data['streamflow_index']
    site_names = synthetic_data['streamflow_columns']
    
    # Get first realization
    first_realization = streamflow[0, :, :]  # (n_months, n_sites)
    
    # Monthly correlation for synthetic
    monthly_corr_path = plot_dir / f"{filter_name}_{basin_name.lower()}_synthetic_monthly_correlation.png"
    plot_correlation_matrix(first_realization, 
                           f'Synthetic Streamflow - {basin_name} ({filter_name}) - First Realization',
                           monthly_corr_path, 
                           data_type='monthly')
    
    # Annual correlation for synthetic
    n_months, n_sites = first_realization.shape
    n_years = n_months // 12
    annual_synthetic = first_realization.reshape(n_years, 12, n_sites).sum(axis=1)  # (n_years, n_sites)
    annual_corr_path = plot_dir / f"{filter_name}_{basin_name.lower()}_synthetic_annual_correlation.png"
    plot_correlation_matrix(annual_synthetic,
                           f'Synthetic Streamflow - {basin_name} ({filter_name}) - First Realization',
                           annual_corr_path,
                           data_type='annual')
    
    # 4. Correlation plots for historical data
    # Monthly correlation for historical
    hist_monthly_corr_path = plot_dir / f"{filter_name}_{basin_name.lower()}_historical_monthly_correlation.png"
    plot_correlation_matrix(hist_monthly.values,
                           f'Historical Streamflow - {basin_name}',
                           hist_monthly_corr_path,
                           data_type='monthly')
    
    # Annual correlation for historical
    hist_annual = hist_monthly.resample('Y').sum()
    hist_annual_corr_path = plot_dir / f"{filter_name}_{basin_name.lower()}_historical_annual_correlation.png"
    plot_correlation_matrix(hist_annual.values,
                           f'Historical Streamflow - {basin_name}',
                           hist_annual_corr_path,
                           data_type='annual')
    
    print(f"All plots saved to {plot_dir}")

def main():
    parser = argparse.ArgumentParser(description='Explore synthetic streamflow data and generate plots')
    parser.add_argument('--filter', help='Filter name to process (e.g., basic, cooler, hotter)')
    parser.add_argument('--basin', help='Basin name to process (e.g., Colorado, Trinity, Brazos)')
    args = parser.parse_args()
    
    # Load configurations
    with open(basins_path, "r") as f:
        BASINS = json.load(f)
    
    with open(ensemble_filters_path, "r") as f:
        ENSEMBLE_CONFIG = json.load(f)
    
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
    for filter_set in filter_sets:
        filter_name = filter_set["name"]
        
        print(f"\nProcessing filter: {filter_name}")
        
        for basin_name, basin in basins.items():
            explore_streamflow(basin_name, basin, filter_name)

if __name__ == "__main__":
    main()



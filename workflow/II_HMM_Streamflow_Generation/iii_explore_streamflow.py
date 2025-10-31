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

def plot_annual_streamflow_with_historical(synthetic_data, hist_monthly, gage_name, basin_name, filter_name, output_path):
    """Plot annual streamflow for outflow gage: 10 random synthetic realizations plus historical data."""
    streamflow = synthetic_data['streamflow']  # (n_realizations, n_months, n_sites)
    time_index = pd.to_datetime(synthetic_data['streamflow_index'])
    site_names = synthetic_data['streamflow_columns']
    
    # Find the outflow gage index
    gage_index = list(site_names).index(gage_name)
    
    # Determine number of realizations to plot
    n_realizations = streamflow.shape[0]
    n_to_plot = min(10, n_realizations)
    
    # Select random realizations
    np.random.seed(42)
    realization_indices = np.random.choice(n_realizations, size=n_to_plot, replace=False)
    
    # Get dimensions
    n_months = streamflow.shape[1]
    n_years = n_months // 12
    
    # Get years from synthetic time index
    synthetic_years = np.arange(int(time_index[0].year), int(time_index[0].year) + n_years)
    
    # Calculate historical annual sums
    hist_annual = hist_monthly[gage_name].resample('Y').sum()
    hist_years = hist_annual.index.year
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot synthetic realizations for outflow gage only
    for real_idx in realization_indices:
        realization_data = streamflow[real_idx, :, gage_index]  # (n_months,)
        streamflow_reshaped = realization_data.reshape(n_years, 12)
        annual_streamflow = streamflow_reshaped.sum(axis=1)  # (n_years,)
        ax.plot(synthetic_years, annual_streamflow, linewidth=1.5, alpha=0.3, color='blue')
    
    # Plot historical data for outflow gage
    ax.plot(hist_years, hist_annual.values, linewidth=2, alpha=0.8, color='red')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=1.5, alpha=0.5, label='Synthetic (10 realizations)'),
        Line2D([0], [0], color='red', linewidth=2, alpha=0.8, label='Historical')
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='best')
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Annual Streamflow (acre-feet)', fontsize=12)
    ax.set_title(f'Annual Streamflow at Outflow Gage - {basin_name} ({filter_name})\nGage: {gage_name}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved annual streamflow plot to {output_path}")

def plot_correlation_matrix(streamflow, site_names, title, output_path):
    """Create correlation matrix heatmap."""
    correlation_matrix = np.corrcoef(streamflow.T)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cmap = plt.get_cmap("viridis")
    im = ax.matshow(correlation_matrix, cmap=cmap, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation plot to {output_path}")

def explore_streamflow(basin_name, basin, filter_name):
    """Generate all exploratory plots for a basin/filter combination."""
    print(f"\nExploring streamflow for {basin_name} ({filter_name})")
    
    # Create output directory for plots
    plot_dir = output_dir / f"{filter_name}" / f"{basin_name.lower()}" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Load synthetic data
    synthetic_data = load_synthetic_data(basin_name, filter_name)
    if synthetic_data is None:
        return None
    
    # Load historical data
    hist_monthly = load_historical_data(basin)
    
    # Get outflow gage name
    gage_name = basin["gage_name"]
    
    streamflow = synthetic_data['streamflow']  # (n_realizations, n_months, n_sites)
    site_names = synthetic_data['streamflow_columns']
    
    # Select a random realization for correlation analysis
    np.random.seed(42)
    random_realization_idx = np.random.randint(0, streamflow.shape[0])
    random_realization = streamflow[random_realization_idx, :, :]  # (n_months, n_sites)
    
    # 1. Historical monthly gage correlation plot
    hist_monthly_corr_path = plot_dir / f"{filter_name}_{basin_name.lower()}_historical_monthly_correlation.png"
    plot_correlation_matrix(hist_monthly.values, 
                           hist_monthly.columns.tolist(),
                           f'Historical Monthly Streamflow Correlation - {basin_name}',
                           hist_monthly_corr_path)
    
    # 2. Synthetic monthly gage correlation plot for a single random realization
    synthetic_monthly_corr_path = plot_dir / f"{filter_name}_{basin_name.lower()}_synthetic_monthly_correlation.png"
    plot_correlation_matrix(random_realization,
                           site_names,
                           f'Synthetic Monthly Streamflow Correlation - {basin_name} ({filter_name})',
                           synthetic_monthly_corr_path)
    
    # 3. Annual sums over time for 10 random realizations and historical data (outflow gage only)
    annual_streamflow_path = plot_dir / f"{filter_name}_{basin_name.lower()}_annual_streamflow.png"
    plot_annual_streamflow_with_historical(synthetic_data, hist_monthly, gage_name, basin_name, filter_name, annual_streamflow_path)
    
    print(f"All plots saved to {plot_dir}")
    
    return synthetic_data

def calculate_streamflow_statistics(synthetic_data, gage_name):
    """Calculate summary statistics for annual streamflow at the outflow gage."""
    streamflow = synthetic_data['streamflow']  # (n_realizations, n_months, n_sites)
    site_names = synthetic_data['streamflow_columns']
    
    # Find the outflow gage index
    gage_index = list(site_names).index(gage_name)
    
    # Get data for outflow gage only
    gage_data = streamflow[:, :, gage_index]  # (n_realizations, n_months)
    
    # Convert to annual sums
    n_realizations, n_months = gage_data.shape
    n_years = n_months // 12
    
    # Reshape to (n_realizations, n_years, 12) and sum over months
    annual_data = gage_data[:, :n_years*12].reshape(n_realizations, n_years, 12).sum(axis=2)  # (n_realizations, n_years)
    
    # Flatten to get all annual values across realizations and years
    annual_values = annual_data.flatten()
    
    stats = {
        'mean': np.mean(annual_values),
        'median': np.median(annual_values),
        'min': np.min(annual_values),
        'max': np.max(annual_values),
        'std': np.std(annual_values)
    }
    
    return stats

def round_to_n_digits(x, n=4):
    """Round a number to n significant digits."""
    if x == 0:
        return 0
    from math import log10, floor
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))

def generate_basin_summary_table(basin_name, basin, filter_sets):
    """Generate summary table of streamflow statistics across all filter types for a basin (outflow gage only)."""
    print(f"\n{'='*60}")
    print(f"Generating summary statistics for {basin_name}")
    print(f"{'='*60}")
    
    # Get outflow gage name
    gage_name = basin["gage_name"]
    
    # Create output directory
    basin_summary_dir = output_dir / f"basin_summaries"
    basin_summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect statistics for each filter
    rows = []
    
    for filter_set in filter_sets:
        filter_name = filter_set["name"]
        print(f"Processing filter: {filter_name}")
        
        synthetic_data = load_synthetic_data(basin_name, filter_name)
        if synthetic_data is None:
            print(f"  Skipping {filter_name} - data not found")
            continue
        
        stats = calculate_streamflow_statistics(synthetic_data, gage_name)
        
        row = {
            'Filter': filter_name,
            'Mean': stats['mean'],
            'Median': stats['median'],
            'Min': stats['min'],
            'Max': stats['max'],
            'Std': stats['std']
        }
        rows.append(row)
    
    if not rows:
        print(f"No data found for basin {basin_name}")
        return
    
    # Create summary table
    summary_df = pd.DataFrame(rows)
    
    # Save CSV
    output_path = basin_summary_dir / f"{basin_name.lower()}_summary.csv"
    summary_df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"Saved summary table to {output_path}")
    
    # Create version with 4 significant digits for LaTeX
    summary_df_latex = summary_df.copy()
    for col in ['Mean', 'Median', 'Min', 'Max', 'Std']:
        summary_df_latex[col] = summary_df_latex[col].apply(lambda x: round_to_n_digits(x, 4))
    
    # Save LaTeX table
    latex_output_path = basin_summary_dir / f"{basin_name.lower()}_summary.tex"
    latex_str = summary_df_latex.to_latex(index=False, escape=False, float_format='%.0f')
    with open(latex_output_path, 'w') as f:
        f.write(latex_str)
    print(f"Saved LaTeX table to {latex_output_path}")
    print(f"  Outflow gage: {gage_name}")

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
    
    # Process selected combinations - generate plots for each basin/filter
    for filter_set in filter_sets:
        filter_name = filter_set["name"]
        
        print(f"\nProcessing filter: {filter_name}")
        
        for basin_name, basin in basins.items():
            explore_streamflow(basin_name, basin, filter_name)
    
    # Generate summary tables for each basin across all filters
    print("\n" + "="*60)
    print("GENERATING BASIN SUMMARY STATISTICS")
    print("="*60)
    
    for basin_name, basin in basins.items():
        generate_basin_summary_table(basin_name, basin, filter_sets)

if __name__ == "__main__":
    main()



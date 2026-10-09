"""
Loads the finalized (archived) datasets and produces end-of-pipeline diagnostics:
streamflow plots at the outlet gage (mirroring II_HMM_Streamflow_Generation/iii_explore_streamflow.py,
but read from the compressed archive files) and per-basin summary tables of streamflow
and WRAP statistics for the paper. WRAP-variable plots live in III_WRAP_Execution/iv_explore_wrap_outputs.py.
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib.lines import Line2D

from toolkit.wrap.io import flo_to_df
from toolkit.wrap.processing import aggregate_over_entities
from toolkit.hmm.metrics import compute_drought_metrics_ensemble
from toolkit.graphics.hmm import plot_drought_metrics
from toolkit.utils.workflow_cli import parse_filter_basin_args, select_filter_sets_and_basins
from toolkit.paths import archived_dataset_path
from toolkit import repo_data_path, outputs_path

sns.set_style("whitegrid")

basins_path = repo_data_path / "configs" / "basins.json"
ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters.json"
# Plots and summary tables go here, not into the archive folder itself, so
# outputs/data_archive stays exactly what gets published.
output_dir = outputs_path / "final_exploration"

N_REALIZATIONS_TO_PLOT = 10
RANDOM_SEED = 42

### Functions ###

def load_final_dataset(basin_name, filter_name):
    """Load the archived, finalized NetCDF (streamflow + WRAP outputs)."""
    nc_path = archived_dataset_path(filter_name, basin_name)

    if not nc_path.exists():
        print(f"Final dataset not found at {nc_path}")
        return None

    return xr.open_dataset(nc_path)

def load_historical_data(basin):
    """Load historical streamflow data from FLO file."""
    flo_file = repo_data_path / basin["flo_file"]
    hist_monthly = flo_to_df(str(flo_file))
    return hist_monthly

def plot_correlation_matrix(data_2d, title, output_path, vmin=0, vmax=1):
    """Create correlation matrix heatmap. data_2d: (n_time, n_entities).

    Some WRAP variables (e.g. shortage_ratio) legitimately have zero-variance
    columns for a given realization, like a water right that never sees shortage,
    which makes correlation mathematically undefined (NaN) for that entity.
    That's a correct answer, not a bug, so the resulting divide-by-zero warning
    is suppressed rather than fixed."""
    with np.errstate(invalid='ignore'):
        correlation_matrix = np.corrcoef(data_2d.T)

    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = plt.get_cmap("viridis")
    im = ax.matshow(correlation_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=12)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved correlation plot to {output_path}")

def plot_annual_streamflow_with_historical(ds, hist_monthly, gage_name, basin_name, filter_name, output_path):
    """Plot annual streamflow for outflow gage: N random realizations plus historical data."""
    streamflow = ds['synthetic_streamflow'].values  # (n_realizations, n_months, n_gages)
    time_index = pd.to_datetime(ds['time_step'].values)
    gage_ids = list(ds['gage_id'].values)

    gage_index = gage_ids.index(gage_name)

    n_realizations = streamflow.shape[0]
    n_to_plot = min(N_REALIZATIONS_TO_PLOT, n_realizations)

    rng = np.random.default_rng(RANDOM_SEED)
    realization_indices = rng.choice(n_realizations, size=n_to_plot, replace=False)

    n_months = streamflow.shape[1]
    n_years = n_months // 12

    synthetic_years = np.arange(int(time_index[0].year), int(time_index[0].year) + n_years)

    hist_annual = hist_monthly[gage_name].resample('YS').sum()
    hist_years = hist_annual.index.year

    fig, ax = plt.subplots(figsize=(14, 7))

    for real_idx in realization_indices:
        realization_data = streamflow[real_idx, :, gage_index]
        streamflow_reshaped = realization_data.reshape(n_years, 12)
        annual_streamflow = streamflow_reshaped.sum(axis=1)
        ax.plot(synthetic_years, annual_streamflow, linewidth=1.5, alpha=0.3, color='blue')

    ax.plot(hist_years, hist_annual.values, linewidth=2, alpha=0.8, color='red')

    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=1.5, alpha=0.5, label=f'Synthetic ({n_to_plot} realizations)'),
        Line2D([0], [0], color='red', linewidth=2, alpha=0.8, label='Historical')
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='best')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Annual Streamflow (acre-feet)', fontsize=12)
    ax.set_title(f'Annual Streamflow - {basin_name} ({filter_name})\nGage: {gage_name}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved annual streamflow plot to {output_path}")

def plot_monthly_climatology(ds, hist_monthly, gage_name, basin_name, filter_name, output_path,
                              percentile_band=(10, 90)):
    """Plot monthly climatology (mean value per calendar month) at the outflow gage:
    a percentile band + median across *all* realizations, historical mean overlaid.
    Unlike the spaghetti-line annual plots this doesn't subsample realizations; a
    climatology curve per realization is cheap enough not to need it.
    """
    streamflow = ds['synthetic_streamflow'].values  # (n_realizations, n_months, n_gages)
    gage_ids = list(ds['gage_id'].values)
    gage_index = gage_ids.index(gage_name)

    n_realizations, n_months = streamflow.shape[0], streamflow.shape[1]
    n_years = n_months // 12

    # (n_realizations, n_years, 12) -> mean across years -> (n_realizations, 12):
    # one climatological monthly mean per realization.
    gage_data = streamflow[:, :n_years * 12, gage_index].reshape(n_realizations, n_years, 12)
    climatology = gage_data.mean(axis=1)

    months = np.arange(1, 13)
    median = np.median(climatology, axis=0)
    lower = np.percentile(climatology, percentile_band[0], axis=0)
    upper = np.percentile(climatology, percentile_band[1], axis=0)

    hist_climatology = hist_monthly[gage_name].groupby(hist_monthly.index.month).mean().reindex(months)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(months, lower, upper, color='blue', alpha=0.2,
                     label=f'Synthetic {percentile_band[0]}-{percentile_band[1]}th percentile')
    ax.plot(months, median, color='blue', linewidth=2, alpha=0.8, label='Synthetic median')
    ax.plot(months, hist_climatology.values, color='red', linewidth=2, alpha=0.8, label='Historical mean')

    ax.set_xticks(months)
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Mean Monthly Streamflow (acre-feet)', fontsize=12)
    ax.set_title(f'Streamflow Climatology - {basin_name} ({filter_name})\nGage: {gage_name}',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved monthly climatology plot to {output_path}")

def explore_final_dataset(basin_name, basin, filter_name):
    """Generate streamflow diagnostic plots from the archived dataset for a basin/filter combination."""
    print(f"\nExploring final dataset for {basin_name} ({filter_name})")

    plot_dir = output_dir / basin_name / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    ds = load_final_dataset(basin_name, filter_name)
    if ds is None:
        return None

    hist_monthly = load_historical_data(basin)
    gage_name = basin["gage_name"]

    streamflow = ds['synthetic_streamflow'].values  # (n_realizations, n_months, n_gages)
    gage_ids = list(ds['gage_id'].values)

    rng = np.random.default_rng(RANDOM_SEED)
    random_realization_idx = rng.integers(0, streamflow.shape[0])

    ### Streamflow plots (mirrors iii_explore_streamflow.py) ###

    # Fixed CPs are spliced in verbatim from history for every realization (see
    # toolkit.utils.fixed_control_points). They're identical to, or for the
    # historical matrix literally are, the historical record by construction, so
    # they trivially correlate at ~1 and just add noise to these matrices.
    fixed_cps = set(basin.get("fixed_control_points", []))
    free_gage_mask = np.array([g not in fixed_cps for g in gage_ids])
    free_gage_ids = [g for g in gage_ids if g not in fixed_cps]

    hist_monthly_corr_path = plot_dir / f"{filter_name}_{basin_name.lower()}_historical_monthly_correlation.png"
    plot_correlation_matrix(hist_monthly[free_gage_ids].values,
                           f'Historical Streamflow Correlation - {basin_name}',
                           hist_monthly_corr_path)

    synthetic_monthly_corr_path = plot_dir / f"{filter_name}_{basin_name.lower()}_synthetic_monthly_correlation.png"
    plot_correlation_matrix(streamflow[random_realization_idx][:, free_gage_mask],
                           f'Synthetic Streamflow Correlation - {basin_name} ({filter_name})',
                           synthetic_monthly_corr_path)

    annual_streamflow_path = plot_dir / f"{filter_name}_{basin_name.lower()}_annual_streamflow.png"
    plot_annual_streamflow_with_historical(ds, hist_monthly, gage_name, basin_name, filter_name, annual_streamflow_path)

    climatology_path = plot_dir / f"{filter_name}_{basin_name.lower()}_monthly_climatology.png"
    plot_monthly_climatology(ds, hist_monthly, gage_name, basin_name, filter_name, climatology_path)

    gage_index = gage_ids.index(gage_name)
    n_realizations, n_months = streamflow.shape[0], streamflow.shape[1]
    n_years = n_months // 12
    annual_ensemble = streamflow[:, :n_years * 12, gage_index].reshape(n_realizations, n_years, 12).sum(axis=2)
    hist_annual = hist_monthly[gage_name].resample('YS').sum().to_numpy()

    metrics_df, historical_metrics = compute_drought_metrics_ensemble(annual_ensemble, hist_annual)
    drought_metrics_dir = plot_dir / "drought_metrics"
    plot_drought_metrics(metrics_df, historical_metrics, drought_metrics_dir)

    print(f"All plots saved to {plot_dir}")

    ds.close()
    return True

def calculate_streamflow_statistics(ds, gage_name):
    """Calculate summary statistics for annual streamflow at the outflow gage."""
    streamflow = ds['synthetic_streamflow'].values
    gage_ids = list(ds['gage_id'].values)
    gage_index = gage_ids.index(gage_name)

    gage_data = streamflow[:, :, gage_index]
    n_realizations, n_months = gage_data.shape
    n_years = n_months // 12

    annual_data = gage_data[:, :n_years*12].reshape(n_realizations, n_years, 12).sum(axis=2)
    annual_values = annual_data.flatten()

    return {
        'mean': np.mean(annual_values),
        'median': np.median(annual_values),
        'min': np.min(annual_values),
        'max': np.max(annual_values),
        'std': np.std(annual_values)
    }

def calculate_wrap_statistics(ds):
    """Calculate basin-wide summary statistics for the WRAP diversion/reservoir outputs.

    No basin-wide aggregate over diversion_or_energy_shortage/shortage_ratio: a
    handful of non-consumptive placeholder rights (bay/estuary inflow requirements,
    dummy accounting rights, e.g. BAY-*, *DUMMY*, TOOL-*) carry deliberately huge
    nominal targets and dominate any right_id-summed or flat-mean total, so a
    single basin-wide number here would misrepresent typical water-user shortage
    rather than describe it. Per-right detail is still visible in the correlation
    and annual time-series plots.
    """
    stats = {}

    if 'reservoir_storage_capacity' in ds.data_vars:
        # Storage capacity is a static reservoir attribute repeated every month,
        # so this is a single basin-wide total, not something to average over time.
        stats['Total Basin Reservoir Storage Capacity (AF)'] = float(
            ds['reservoir_storage_capacity'].isel(realization=0, time_step=0).sum().values
        )

    if 'reservoir_net_evaporation_precipitation_volume' in ds.data_vars:
        n_realizations, n_months, _ = ds['reservoir_net_evaporation_precipitation_volume'].shape
        n_years = n_months // 12

        basin_evap = aggregate_over_entities(ds['reservoir_net_evaporation_precipitation_volume'], 'reservoir_id', 'sum')
        annual_evap = basin_evap[:, :n_years*12].reshape(n_realizations, n_years, 12).sum(axis=2)
        stats['Mean Annual Net Evaporation (AF)'] = np.mean(annual_evap)

    return stats

def round_to_n_digits(x, n=4):
    """Round a number to n significant digits."""
    if x == 0:
        return 0
    from math import log10, floor
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))

def generate_basin_summary_table(basin_name, basin, filter_sets):
    """Generate summary table of streamflow + WRAP statistics across all filter types for a basin."""
    print(f"\n{'='*60}")
    print(f"Generating final summary statistics for {basin_name}")
    print(f"{'='*60}")

    gage_name = basin["gage_name"]

    basin_summary_dir = output_dir / "basin_summaries"
    basin_summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for filter_set in filter_sets:
        filter_name = filter_set["name"]
        print(f"Processing filter: {filter_name}")

        ds = load_final_dataset(basin_name, filter_name)
        if ds is None:
            print(f"  Skipping {filter_name} - data not found")
            continue

        stats = calculate_streamflow_statistics(ds, gage_name)
        wrap_stats = calculate_wrap_statistics(ds)

        row = {
            'Filter': filter_name,
            'Mean': stats['mean'],
            'Median': stats['median'],
            'Min': stats['min'],
            'Max': stats['max'],
            'Std': stats['std'],
            **wrap_stats,
        }
        rows.append(row)
        ds.close()

    if not rows:
        print(f"No data found for basin {basin_name}")
        return

    summary_df = pd.DataFrame(rows)

    output_path = basin_summary_dir / f"{basin_name.lower()}_final_summary.csv"
    summary_df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"Saved summary table to {output_path}")

    summary_df_latex = summary_df.copy()
    numeric_cols = [c for c in summary_df.columns if c != 'Filter']
    for col in numeric_cols:
        summary_df_latex[col] = summary_df_latex[col].apply(lambda x: round_to_n_digits(x, 4))

    latex_output_path = basin_summary_dir / f"{basin_name.lower()}_final_summary.tex"
    latex_str = summary_df_latex.to_latex(index=False, escape=False, float_format='%.0f')
    with open(latex_output_path, 'w') as f:
        f.write(latex_str)
    print(f"Saved LaTeX table to {latex_output_path}")
    print(f"  Outflow gage: {gage_name}")

### Main ###

def main():
    args = parse_filter_basin_args('Explore the final (archived) synthetic dataset and generate plots')

    with open(basins_path, "r") as f:
        BASINS = json.load(f)

    with open(ensemble_filters_path, "r") as f:
        ENSEMBLE_CONFIG = json.load(f)

    filter_sets, basins = select_filter_sets_and_basins(BASINS, ENSEMBLE_CONFIG, args.filter, args.basin)

    for filter_set in filter_sets:
        filter_name = filter_set["name"]

        print(f"\nProcessing filter: {filter_name}")

        for basin_name, basin in basins.items():
            explore_final_dataset(basin_name, basin, filter_name)

    print("\n" + "="*60)
    print("GENERATING FINAL BASIN SUMMARY STATISTICS")
    print("="*60)

    for basin_name, basin in basins.items():
        generate_basin_summary_table(basin_name, basin, filter_sets)

if __name__ == "__main__":
    main()

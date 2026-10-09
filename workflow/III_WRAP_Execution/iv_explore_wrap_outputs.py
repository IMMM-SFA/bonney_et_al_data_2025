"""
Diagnostic plots for the WRAP outputs in the WRAP-augmented datasets. For each basin-wide
WRAP variable in VARIABLES: a per-entity correlation matrix (one random realization and the
historical baseline), annual time series of random realizations, and a monthly climatology
band, each against a single WRAP run of the historical record.
"""
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from toolkit.wrap.io import out_to_dfs
from toolkit.wrap.processing import process_diversion_csv, process_reservoir_csv, aggregate_over_entities
from toolkit.wrap.execution_slot import LocalWRAPExecutionSlot
from toolkit.utils.workflow_cli import parse_filter_basin_args, select_filter_sets_and_basins
from toolkit.paths import wrap_augmented_dataset_path
from toolkit import repo_data_path, outputs_path

sns.set_style("whitegrid")

### Settings ###
# Must match the DAT_SUFFIX the ensemble was run with (ii_execute_wrap.py) so the historical
# baseline reflects the same WAM scenario.
DAT_SUFFIX = "_initial_storage_median_historical.dat"
N_REALIZATIONS_TO_PLOT = 10
RANDOM_SEED = 42

# Basin-wide WRAP variables to plot: how to aggregate over rights/reservoirs, and axis labels.
VARIABLES = {
    "shortage_ratio": dict(agg="mean", label="Mean shortage ratio", short="shortage_ratio"),
    "reservoir_net_evaporation_precipitation_volume": dict(agg="sum", label="Net evaporation (acre-feet)", short="reservoir_evaporation"),
}

### Path Configuration ###
basins_path = repo_data_path / "configs" / "basins.json"
ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters.json"
WRAP_SIM_PATH = repo_data_path / "WRAP" / "SIM.exe"
output_dir = outputs_path / "wrap_results"
HISTORICAL_WRAP_DIR = output_dir / "_historical_wrap_run"

### Functions ###

def run_historical_wrap(basin_name, basin):
    """One WRAP run on the historical FLO record, giving the historical baseline of each
    VARIABLES entry as a (months x entities) DataFrame. Cached to CSV, keyed by DAT_SUFFIX."""
    cache_dir = HISTORICAL_WRAP_DIR / basin_name / DAT_SUFFIX.strip("_.").removesuffix(".dat")
    caches = {var: cache_dir / f"{var}.csv" for var in VARIABLES}
    if all(p.exists() for p in caches.values()):
        return {var: pd.read_csv(p, index_col=0, parse_dates=True) for var, p in caches.items()}

    flo_file = repo_data_path / basin["flo_file"]
    slot = LocalWRAPExecutionSlot(HISTORICAL_WRAP_DIR / basin_name, flo_file.parent, WRAP_SIM_PATH, flo_file.stem, dat_suffix=DAT_SUFFIX)
    slot.teardown(); slot.setup()
    print(f"Running WRAP on the historical record for {basin_name} (cached after)...")
    flo_name = slot.run(flo_file)
    dfs = out_to_dfs(slot.slot_dir / f"{flo_name}.OUT")
    slot.teardown()

    diversions = process_diversion_csv(dfs["diversions"], column_names=["diversion_or_energy_shortage", "diversion_or_energy_target"], compute_shortage_ratio=True)
    reservoirs = process_reservoir_csv(dfs["reservoirs"], column_names=["reservoir_net_evaporation_precipitation_volume"])
    baselines = {**diversions, **reservoirs}

    cache_dir.mkdir(parents=True, exist_ok=True)
    for var, p in caches.items():
        baselines[var].to_csv(p)
    return {var: baselines[var] for var in VARIABLES}


def basin_series(df, agg):
    """Aggregate a (months x entities) DataFrame to one basin-wide monthly series."""
    return df.sum(axis=1) if agg == "sum" else df.mean(axis=1)


def plot_correlation_matrix(data_2d, title, output_path):
    """Correlation heatmap across entities. Zero-variance columns (e.g. a right that never
    sees shortage) give NaN correlations, which is correct, so the warning is silenced."""
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(data_2d.T)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.matshow(corr, cmap="viridis", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close()


def plot_annual(synthetic, historical, agg, label, title, output_path):
    """Annual basin-wide values of N random realizations with the historical baseline.
    synthetic: (realizations, months) array; historical: monthly Series."""
    n_real, n_months = synthetic.shape
    n_years = n_months // 12
    years = historical.index.year[::12][:n_years]
    reduce = (lambda a: a.sum(axis=1)) if agg == "sum" else (lambda a: a.mean(axis=1))

    fig, ax = plt.subplots(figsize=(14, 7))
    picks = np.random.default_rng(RANDOM_SEED).choice(n_real, size=min(N_REALIZATIONS_TO_PLOT, n_real), replace=False)
    for i, idx in enumerate(picks):
        ax.plot(years, reduce(synthetic[idx, :n_years * 12].reshape(n_years, 12)), lw=1.5, alpha=0.4,
                color="darkorange", label=f"Synthetic ({len(picks)} realizations)" if i == 0 else None)
    hist_annual = historical.resample("YS").sum() if agg == "sum" else historical.resample("YS").mean()
    ax.plot(hist_annual.index.year, hist_annual.values, lw=2, alpha=0.8, color="red", label="Historical")
    ax.set_xlabel("Year", fontsize=12); ax.set_ylabel(label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold"); ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close()


def plot_climatology(synthetic, historical, label, title, output_path, band=(10, 90)):
    """Mean value per calendar month: 10-90% band and median across realizations, with the
    historical baseline. synthetic: (realizations, months) array; historical: monthly Series."""
    n_real, n_months = synthetic.shape
    n_years = n_months // 12
    clim = synthetic[:, :n_years * 12].reshape(n_real, n_years, 12).mean(axis=1)  # (realizations, 12)
    months = np.arange(1, 13)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(months, np.percentile(clim, band[0], axis=0), np.percentile(clim, band[1], axis=0),
                    color="darkorange", alpha=0.2, label=f"Synthetic {band[0]}-{band[1]}th percentile")
    ax.plot(months, np.median(clim, axis=0), color="darkorange", lw=2, alpha=0.8, label="Synthetic median")
    ax.plot(months, historical.groupby(historical.index.month).mean().reindex(months).values,
            color="red", lw=2, alpha=0.8, label="Historical mean")
    ax.set_xticks(months); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlabel("Month", fontsize=12); ax.set_ylabel(label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold"); ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close()


def explore_wrap_outputs(basin_name, basin, filter_name):
    """All WRAP diagnostic plots for one basin/filter combination."""
    print(f"\nExploring WRAP outputs for {basin_name} ({filter_name})")
    nc_path = wrap_augmented_dataset_path(filter_name, basin_name)
    if not nc_path.exists():
        print(f"WRAP-augmented dataset not found at {nc_path}")
        return
    ds = xr.open_dataset(nc_path)
    plot_dir = output_dir / filter_name / basin_name / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{filter_name}_{basin_name.lower()}"
    suffix = f"{basin_name} ({filter_name})"

    baselines = run_historical_wrap(basin_name, basin)
    example = np.random.default_rng(RANDOM_SEED).integers(0, ds.sizes["realization"])

    for var, spec in VARIABLES.items():
        if var not in ds:
            print(f"  {var} not in dataset, skipping")
            continue
        entity_dim = "right_id" if "right_id" in ds[var].dims else "reservoir_id"
        synthetic = aggregate_over_entities(ds[var], entity_dim, spec["agg"])  # (realizations, months)
        historical = basin_series(baselines[var], spec["agg"])

        plot_correlation_matrix(baselines[var].values, f"Historical {spec['label']} correlation - {basin_name}",
                                plot_dir / f"{prefix}_historical_{spec['short']}_correlation.png")
        plot_correlation_matrix(ds[var].isel(realization=example).values, f"Synthetic {spec['label']} correlation - {suffix}",
                                plot_dir / f"{prefix}_synthetic_{spec['short']}_correlation.png")
        plot_annual(synthetic, historical, spec["agg"], spec["label"], f"Annual {spec['label']} - {suffix}",
                    plot_dir / f"{prefix}_annual_{spec['short']}.png")
        plot_climatology(synthetic, historical, spec["label"], f"{spec['label']} climatology - {suffix}",
                         plot_dir / f"{prefix}_{spec['short']}_climatology.png")
    ds.close()
    print(f"Plots saved to {plot_dir}")

### Main ###

def main():
    args = parse_filter_basin_args("Explore WRAP outputs (diversions/reservoirs) and generate plots")
    with open(basins_path) as f:
        BASINS = json.load(f)
    with open(ensemble_filters_path) as f:
        ENSEMBLE_CONFIG = json.load(f)
    filter_sets, basins = select_filter_sets_and_basins(BASINS, ENSEMBLE_CONFIG, args.filter, args.basin)
    for filter_set in filter_sets:
        for basin_name, basin in basins.items():
            explore_wrap_outputs(basin_name, basin, filter_set["name"])

if __name__ == "__main__":
    main()

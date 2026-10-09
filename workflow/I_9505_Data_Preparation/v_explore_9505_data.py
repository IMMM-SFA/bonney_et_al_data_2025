"""
Diagnostics for the 9505 preparation stage: for each basin and ensemble filter, compares
the distribution of annual flow at the basin outlet in the processed 9505 data (the HMM
training data) against the historical WRAP record at the same gage.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

from toolkit.data.ninetyfiveofive import load_doe_data, load_historical_data
from toolkit.utils.workflow_cli import parse_filter_basin_args, select_filter_sets_and_basins
from toolkit import repo_data_path, outputs_path

### Settings ###
PERIOD = "2020_2059"  # 9505 period to compare (must match i_train_hmm_models.py)
LOG_TRANSFORM = True

### Path Configuration ###
basins_path = repo_data_path / "configs" / "basins.json"
ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters.json"
nc_file_path = outputs_path / "9505" / "reach_subset_combined" / f"master_streamflow_{PERIOD}_af.nc"
output_dir = outputs_path / "9505" / "plots"

### Functions ###

def plot_annual_distribution(basin_name, basin, ensemble_filters, filter_name):
    """Histogram of (log) annual outlet flow: historical record vs. the filtered 9505 members."""
    gage_name, reach_id = basin["gage_name"], basin["reach_id"]
    flo_file = repo_data_path / basin["flo_file"]

    hist_data, _ = load_historical_data(flo_file=flo_file, gage_name=gage_name, reach_id=reach_id,
                                        aggregate_annually=True, log1p_transform=LOG_TRANSFORM)
    doe_data, _ = load_doe_data(nc_file=nc_file_path, flo_file=flo_file, gage_name=gage_name, reach_id=reach_id,
                                period=PERIOD, aggregate_annually=True, log1p_transform=LOG_TRANSFORM,
                                ensemble_filters=ensemble_filters)
    hist = np.asarray(hist_data.values if hasattr(hist_data, "values") else hist_data).ravel()
    doe = np.asarray(doe_data)

    hist_mean, doe_mean = hist.mean(), doe.mean()
    label = (lambda m: f"{np.expm1(m):,.0f} AF") if LOG_TRANSFORM else (lambda m: f"{m:,.0f} AF")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(hist, bins=15, alpha=0.7, density=True, color="tab:blue", label=f"Historical ({len(hist)} yrs)")
    ax.hist(doe.ravel(), bins=15, alpha=0.7, density=True, color="tab:green",
            label=f"9505 {PERIOD.replace('_', '-')} ({doe.shape[0]} members x {doe.shape[1]} yrs)")
    for m, c, name in [(hist_mean, "tab:blue", "Historical"), (doe_mean, "tab:green", "9505")]:
        ax.axvline(m, color="white", lw=5); ax.axvline(m, color=c, lw=3, label=f"{name} mean: {label(m)}")
    ax.set_title(f"Annual outlet flow: {basin_name}, gage {gage_name}, reach {reach_id}, filter {filter_name}")
    ax.set_xlabel("log(annual flow + 1)" if LOG_TRANSFORM else "annual flow (acre-feet)")
    ax.set_ylabel("Density"); ax.legend()
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{filter_name}_{basin_name.lower()}_annual_flow_9505_vs_historical.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")

### Main ###

def main():
    args = parse_filter_basin_args("Compare processed 9505 annual flows against the historical record")

    with open(basins_path, "r") as f:
        BASINS = json.load(f)
    with open(ensemble_filters_path, "r") as f:
        ENSEMBLE_CONFIG = json.load(f)

    filter_sets, basins = select_filter_sets_and_basins(BASINS, ENSEMBLE_CONFIG, args.filter, args.basin)
    for filter_set in filter_sets:
        for basin_name, basin in basins.items():
            plot_annual_distribution(basin_name, basin, filter_set["filters"], filter_set["name"])

if __name__ == "__main__":
    main()

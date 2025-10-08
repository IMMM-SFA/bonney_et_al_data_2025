"""
This script trains Bayesian Hidden Markov Models (HMM) on the 9505 data.
"""
import numpy as np
from pathlib import Path
import json
import argparse
import matplotlib.pyplot as plt
from toolkit.hmm.model import BayesianStreamflowHMM
from toolkit.data.ninetyfiveofive import load_doe_data, load_historical_data
from toolkit.hmm.utils import generate_prior_config_from_historical
from toolkit import repo_data_path, outputs_path
import arviz as az


### Settings ###
FORCE_RECOMPUTE = True # Whether to recompute the model if it already exists
LOG_TRANSFORM = True # Whether to log transform the data
PERIOD = "2020_2059" # Time period of 9505 data used for training

### Path Configuration ###
basins_path = repo_data_path / "configs" / "basins.json"
ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters_basic.json"
# ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters.json"
nc_file_path = outputs_path / "9505" / "reach_subset_combined" / f"master_streamflow_{PERIOD}_af.nc"

### Functions ###

def train_basin_hmm(basin_name, basin, ensemble_filters, filter_name):
    """Train HMM for a single basin with a specific set of ensemble filters."""
    if basin_name == "Rio Grande":
        return
    
    gage_name = basin["gage_name"]
    reach_id = basin["reach_id"]
    flo_file = repo_data_path / basin["flo_file"]
    output_dir = outputs_path / "bayesian_hmm" / filter_name / f"{basin_name.lower().replace(' ', '_')}_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load historical data separately
    hist_data, hist_metadata = load_historical_data(
        flo_file=flo_file,
        gage_name=gage_name,
        reach_id=reach_id,
        aggregate_annually=True,
        log1p_transform=LOG_TRANSFORM
    )

    # Load and prepare future training data
    doe_data, doe_metadata = load_doe_data(
        nc_file=nc_file_path,
        flo_file=flo_file,
        gage_name=gage_name,
        reach_id=reach_id,
        period=PERIOD,
        aggregate_annually=True,
        log1p_transform=LOG_TRANSFORM,
        ensemble_filters=ensemble_filters,
    )

    # Diagnostic plot: Compare historical and future data before model fit
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(hist_data.values, bins=15, alpha=0.7, label='Historical', color='tab:blue', density=True)
    ax.hist(doe_data.flatten(), bins=15, alpha=0.7, label='9505 2020-2059 (all ensembles)', color='tab:green', density=True)
    # Add vertical lines for means with white outline
    hist_mean = np.mean(hist_data.values)
    future_mean = np.mean(doe_data.flatten())
    # White outline first, then colored line on top
    ax.axvline(x=hist_mean, color='white', linestyle='-', linewidth=5, alpha=1.0)
    ax.axvline(x=hist_mean, color='tab:blue', linestyle='-', linewidth=3, alpha=0.9, label=f'Historical Mean ({np.e**hist_mean:.2f})')
    ax.axvline(x=future_mean, color='white', linestyle='-', linewidth=5, alpha=1.0)
    ax.axvline(x=future_mean, color='tab:green', linestyle='-', linewidth=3, alpha=0.9, label=f'Future Mean ({np.e**future_mean:.2f})')
    ax.set_title(f'Histogram: Annual Log-Transformed Flow (Normalized)\n Gage: {gage_name}, Reach: {reach_id}, Filter: {filter_name}')
    ax.set_xlabel('log(Flow + 1)')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'diagnostic_hist_future_vs_hist_{filter_name}.png', dpi=150)
    plt.close()

    # Model path
    model_path = output_dir / f"{gage_name}_{filter_name}_model"

    # Fit or load model
    if not FORCE_RECOMPUTE and (model_path.with_suffix(".nc")).exists():
        return model_path
    
    # Generate priors from historical data
    prior_config = generate_prior_config_from_historical(
        hist_data=hist_data.values,
        n_states=2,
        log1p_transform=LOG_TRANSFORM
    )
    
    # Fit HMM
    model = BayesianStreamflowHMM(
        n_states=2,
        random_seed=42,
        prior_config=prior_config
    )
    fit_params = {
        "data": doe_data,
        "draws": 2000,
        "tune": 2000,
        "chains": 4,
        "target_accept": 0.95,
    }
    model.fit(**fit_params)
    model.save(str(model_path))
    
    # Diagnostics
    rhat = az.rhat(model.idata)
    ess = az.ess(model.idata)
    max_rhat = float(rhat.to_array().max().values)
    min_ess = float(ess.to_array().min().values)
    if max_rhat > 1.1:
        print(f"Model failed to converge! Max R-hat: {max_rhat:.3f}")
        raise RuntimeError("Model failed to converge")
    if min_ess < 100:
        print(f"Low effective sample size: {min_ess:.0f}")
    return model_path

### Main ###

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train HMM models for specific filter-basin combinations')
    parser.add_argument('--filter', help='Filter name to process (e.g., basic, cooler, hotter)')
    parser.add_argument('--basin', help='Basin name to process (e.g., Colorado, Trinity, Brazos)')
    args = parser.parse_args()
    
    # Load basin configuration from JSON
    with open(basins_path, "r") as f:
        BASINS = json.load(f)

    # Load ensemble filters configuration from JSON
    with open(ensemble_filters_path, "r") as f:
        ENSEMBLE_CONFIG = json.load(f)

    # Filter processing based on arguments
    if args.filter:
        filter_sets = [fs for fs in ENSEMBLE_CONFIG["filter_sets"] if fs["name"] == args.filter]
        if not filter_sets:
            print(f"Error: Filter '{args.filter}' not found in configuration")
            return
    else:
        filter_sets = ENSEMBLE_CONFIG["filter_sets"]
    
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
        ensemble_filters = filter_set["filters"]
        
        print(f"Processing filter: {filter_name}")
        
        for basin_name, basin in basins.items():
            print(f"  Training HMM for basin: {basin_name}")
            train_basin_hmm(basin_name, basin, ensemble_filters, filter_name)

if __name__ == "__main__":
    main()

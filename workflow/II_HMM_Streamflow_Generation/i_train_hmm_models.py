"""
This script trains Bayesian Hidden Markov Models (HMM) on the 9505 data.
"""
import numpy as np
import json
from toolkit.hmm.model import BayesianStreamflowHMM
from toolkit.data.ninetyfiveofive import load_doe_data, load_historical_data
from toolkit.hmm.utils import generate_prior_config_from_historical
from toolkit.utils.random_seeds import set_random_seeds, get_seed
from toolkit.utils.workflow_cli import parse_filter_basin_args, select_filter_sets_and_basins
from toolkit.paths import basin_filter_dir
from toolkit import repo_data_path, outputs_path
import arviz as az


### Settings ###
FORCE_RECOMPUTE = True # Whether to recompute the model if it already exists
LOG_TRANSFORM = True # Whether to log transform the data
PERIOD = "2020_2059" # Time period of 9505 data used for training

### Path Configuration ###
basins_path = repo_data_path / "configs" / "basins.json"
ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters.json"
nc_file_path = outputs_path / "9505" / "reach_subset_combined" / f"master_streamflow_{PERIOD}_af.nc"

### Functions ###

def train_basin_hmm(basin_name, basin, ensemble_filters, filter_name):
    """Train HMM for a single basin with a specific set of ensemble filters."""
    
    gage_name = basin["gage_name"]
    reach_id = basin["reach_id"]
    flo_file = repo_data_path / basin["flo_file"]
    output_dir = basin_filter_dir(filter_name, basin_name)
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

    # Model path
    model_path = output_dir / f"{basin_name}_{filter_name}_model"

    # Fit or load model
    if not FORCE_RECOMPUTE and (model_path.with_suffix(".nc")).exists():
        return model_path
    
    # Generate priors from historical data
    prior_config = generate_prior_config_from_historical(
        hist_data=hist_data.values,
        n_states=2,
        log1p_transform=LOG_TRANSFORM
    )
    
    # Set random seeds for reproducible training
    set_random_seeds("hmm_training")
    
    # Fit HMM
    model = BayesianStreamflowHMM(
        n_states=2,
        random_seed=get_seed("hmm_training"),
        prior_config=prior_config
    )
    fit_params = {
        "data": doe_data,
        "draws": 2000,
        "tune": 2000,
        "chains": 4,
        "target_accept": 0.95,
        "sampler": "nuts",
    }
    model.fit(**fit_params)
    model.save(str(model_path))
    
    # Generate comprehensive diagnostics
    print(f"Generating diagnostic plots for {basin_name} - {filter_name}...")
    
    # 1. Basic convergence diagnostics
    rhat = az.rhat(model.idata)
    ess = az.ess(model.idata)
    max_rhat = float(rhat.to_array().max().values)
    min_ess = float(ess.to_array().min().values)
    
    print(f"  Convergence: R-hat max = {max_rhat:.3f}, ESS min = {min_ess:.0f}")
    
    if max_rhat > 1.1:
        print(f"  WARNING: Model failed to converge! Max R-hat: {max_rhat:.3f}")
        # Don't raise error, just warn - let user decide
    if min_ess < 100:
        print(f"  WARNING: Low effective sample size: {min_ess:.0f}")
    
    # Diagnostic plots (MCMC traces, HMM state plots, fit vs. training data) are produced
    # by iii_explore_streamflow.py from the saved model.
    return model_path

### Main ###

def main():
    print(f"Settings: FORCE_RECOMPUTE={FORCE_RECOMPUTE}, LOG_TRANSFORM={LOG_TRANSFORM}, PERIOD={PERIOD!r}")

    # Parse command line arguments
    args = parse_filter_basin_args('Train HMM models for specific filter-basin combinations')

    # Load basin configuration from JSON
    with open(basins_path, "r") as f:
        BASINS = json.load(f)

    # Load ensemble filters configuration from JSON
    with open(ensemble_filters_path, "r") as f:
        ENSEMBLE_CONFIG = json.load(f)

    filter_sets, basins = select_filter_sets_and_basins(BASINS, ENSEMBLE_CONFIG, args.filter, args.basin)

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

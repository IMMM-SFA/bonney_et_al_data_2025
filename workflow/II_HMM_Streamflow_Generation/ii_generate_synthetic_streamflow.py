"""
This script loads trained HMM models and generates synthetic streamflow.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse

from toolkit.hmm.model import BayesianStreamflowHMM
from toolkit.data.ninetyfiveofive import load_historical_data, load_9505_stencil_pool
from toolkit.utils.random_seeds import set_random_seeds, get_seed
from toolkit.utils.fixed_control_points import split_free_and_fixed, splice_fixed_columns
from toolkit import repo_data_path, outputs_path
from toolkit.wrap.io import flo_to_df
from toolkit.data.io import save_netcdf_format, load_netcdf_format


### Settings ###
FORCE_RECOMPUTE = True # Whether to recompute the synthetic streamflow if it already exists
LOG_TRANSFORM = True # Whether to log transform the data
N_ENSEMBLES = 1000 # Number of ensembles to generate

# Which candidate "stencil" pool disaggregation draws monthly shapes from:
#   "historical" - the single observed historical FLO record only (default, matches prior behavior)
#   "9505"       - the DOE 9505 ensemble only, filtered to this basin's HMM training filter
#   "blend"      - historical FLO blocks + 9505 blocks pooled together
STENCIL_SOURCE = "historical"
# Which 9505 period(s) (keys of NINETYFIVEOFIVE_NC_PATHS below) to pool stencils from when
# STENCIL_SOURCE is "9505" or "blend". Multiple periods pool their candidate blocks together.
STENCIL_PERIODS = ["2020_2059"]

# Post-hoc bias correction of the BHMM's raw annual outlet streamflow against the historical
# annual record, applied before disaggregation to monthly (see toolkit.hmm.bias_correction).
# None (default) reproduces prior behavior exactly -- no correction applied.
BIAS_CORRECTION_METHOD = None
BIAS_CORRECTION_KWARGS = {}

### Path Configuration ###
basins_path = repo_data_path / "configs" / "basins.json"
# ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters_basic.json"
ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters.json"

output_dir = outputs_path / "bayesian_hmm"

pcp_reach_mapping_path = outputs_path / "9505" / "pcp_to_reach_mapping.csv"
ninetyfiveofive_nc_dir = outputs_path / "9505" / "reach_subset_combined"
NINETYFIVEOFIVE_NC_PATHS = {
    period: ninetyfiveofive_nc_dir / f"master_streamflow_{period}_af.nc"
    for period in ("1980_2019", "2020_2059", "2060_2099")
}

### Functions ###
def generate_synthetic_streamflow(basin_name, basin, ensemble_filters, filter_name):
    """Generate synthetic streamflow for a single basin using a trained HMM model."""
    
    gage_name = basin["gage_name"]
    reach_id = basin["reach_id"]
    
    # Flo file
    flo_file = repo_data_path / basin["flo_file"]
    
    # Model path
    model_path = output_dir / f"{filter_name}" / f"{basin_name.lower()}" / f"{basin_name}_{filter_name}_model"
    
    # Check if model exists
    if not (model_path.with_suffix(".nc")).exists():
        print(f"Model not found at {model_path}. Please run the training script first.")
        return

    # Load historical data for disaggregation
    # hist_data, hist_metadata = load_historical_data(
    #     flo_file=flo_file,
    #     gage_name=gage_name,
    #     reach_id=reach_id,
    #     aggregate_annually=True,
    #     log1p_transform=LOG_TRANSFORM
    # )
    
    # Historical monthly for disaggregation
    hist_monthly = flo_to_df(str(flo_file))
    site_names = hist_monthly.columns.tolist()

    # fixed_control_points (basins.json) are held constant at their historical values --
    # out-of-basin gages or non-hydrologic placeholder CPs with no real streamflow to
    # synthesize (see toolkit.utils.fixed_control_points). Excluded entirely from HMM
    # disaggregation/stencil-pool construction below, then spliced back in after generation.
    free_sites, fixed_sites = split_free_and_fixed(site_names, basin)
    hist_monthly_free = hist_monthly[free_sites]

    outflow_index = free_sites.index(gage_name)
    num_years = len(hist_monthly) // 12

    # Historical annual outlet streamflow, used as the bias-correction reference when
    # BIAS_CORRECTION_METHOD is set.
    historical_annual = hist_monthly[gage_name].resample("YS").sum().to_numpy()

    # Build the disaggregation stencil pool. "historical" reproduces prior behavior exactly;
    # "9505"/"blend" pull additional candidate monthly shapes from the DOE 9505 ensemble,
    # filtered to the same ensemble_filters this basin's HMM was trained on.
    if STENCIL_SOURCE == "historical":
        stencil_pool = hist_monthly_free.values
    else:
        pcp_reach_mapping = pd.read_csv(pcp_reach_mapping_path)
        doe_stencils = load_9505_stencil_pool(
            site_names=free_sites,
            pcp_reach_mapping=pcp_reach_mapping,
            nc_paths=NINETYFIVEOFIVE_NC_PATHS,
            periods=STENCIL_PERIODS,
            ensemble_filters=ensemble_filters,
        )
        if STENCIL_SOURCE == "9505":
            stencil_pool = doe_stencils
        elif STENCIL_SOURCE == "blend":
            stencil_pool = np.concatenate([hist_monthly_free.values, doe_stencils], axis=0)
        else:
            raise ValueError(f"Unknown STENCIL_SOURCE: {STENCIL_SOURCE!r}")

    # Set random seeds for reproducible generation
    set_random_seeds("hmm_generation")
    
    # Load trained model
    model = BayesianStreamflowHMM.load(str(model_path))

    # Generate synthetic streamflow
    synthetic_h5_path = output_dir / f"{filter_name}" /f"{basin_name.lower()}" / f"{filter_name}_{basin_name.lower()}_synthetic_dataset.nc"

    if os.path.exists(synthetic_h5_path) and not FORCE_RECOMPUTE:
        synthetic_streamflow_dict = load_netcdf_format(synthetic_h5_path)
    else:
        synthetic_streamflow_dict = model.generate_synthetic_streamflow(
            start_year=2020,
            num_years=num_years,
            historical_monthly_data=stencil_pool,
            drought=None,
            random_seed=get_seed("hmm_generation"),
            site_names=free_sites,
            time_index=hist_monthly.index.tolist(),
            h5_path=synthetic_h5_path,
            n_ensembles=N_ENSEMBLES,
            outflow_index=outflow_index,
            bias_correction_method=BIAS_CORRECTION_METHOD,
            historical_annual=historical_annual,
            bias_correction_kwargs=BIAS_CORRECTION_KWARGS,
        )

        # Splice fixed_control_points' historical values back in and restore the full,
        # original column order (generation above only produced the free sites).
        if fixed_sites:
            synthetic_streamflow_dict['streamflow'] = splice_fixed_columns(
                synthetic_streamflow_dict['streamflow'],
                free_sites,
                hist_monthly,
                basin,
                site_names,
            )
            synthetic_streamflow_dict['streamflow_columns'] = site_names

        # Save synthetic streamflow to netcdf
        # Convert ensemble_filters to NetCDF-compatible format (remove None values)
        netcdf_filters = {k: v for k, v in ensemble_filters.items() if v is not None}
        
        global_metadata = {
            'basin_name': basin_name,
            'wrap_outflow_gage': gage_name,
            '9505_reach_id': reach_id,
            'subset_name': filter_name,
            'ensemble_filters': str(netcdf_filters)  # Convert to string for NetCDF compatibility
        }
        save_netcdf_format(synthetic_streamflow_dict, synthetic_h5_path, additional_metadata=global_metadata)

### Main ###

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate synthetic streamflow for specific filter-basin combinations')
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
        ensemble_filters = filter_set["filters"]
        
        print(f"Processing filter: {filter_name}")
        
        for basin_name, basin in basins.items():
            print(f"  Generating synthetic streamflow for basin: {basin_name}")
            generate_synthetic_streamflow(basin_name, basin, ensemble_filters, filter_name)

if __name__ == "__main__":
    main()

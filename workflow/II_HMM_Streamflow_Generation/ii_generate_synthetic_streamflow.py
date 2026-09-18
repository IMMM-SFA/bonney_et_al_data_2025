"""
This script loads trained HMM models and generates synthetic streamflow.
"""
import os
import numpy as np
import pandas as pd
import json

from toolkit.hmm.model import BayesianStreamflowHMM
from toolkit.data.ninetyfiveofive import load_9505_stencil_pool
from toolkit.utils.random_seeds import set_random_seeds, get_seed
from toolkit.utils.fixed_control_points import split_free_and_fixed, splice_fixed_columns
from toolkit.utils.workflow_cli import parse_filter_basin_args, select_filter_sets_and_basins
from toolkit.paths import basin_filter_dir, synthetic_dataset_path
from toolkit import repo_data_path, outputs_path
from toolkit.wrap.io import flo_to_df
from toolkit.data.io import save_netcdf_format, load_netcdf_format


### Settings ###
FORCE_RECOMPUTE = True # Whether to recompute the synthetic streamflow if it already exists
LOG_TRANSFORM = True # Whether to log transform the data
N_ENSEMBLES = 2000 # Number of ensembles to generate

# Disaggregation stencil source: "historical" (default), "9505", or "blend"
STENCIL_SOURCE = "historical"
# 9505 period(s) (keys of NINETYFIVEOFIVE_NC_PATHS) to pool when STENCIL_SOURCE != "historical"
STENCIL_PERIODS = ["2020_2059"]

# Apply toolkit.hmm.bias_correction (stretched-tail quantile mapping) before disaggregation
BIAS_CORRECTION = True

### Path Configuration ###
basins_path = repo_data_path / "configs" / "basins.json"
ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters.json"

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
    flo_file = repo_data_path / basin["flo_file"]
    model_path = basin_filter_dir(filter_name, basin_name) / f"{basin_name}_{filter_name}_model"

    if not (model_path.with_suffix(".nc")).exists():
        print(f"Model not found at {model_path}. Please run the training script first.")
        return

    hist_monthly = flo_to_df(str(flo_file))
    site_names = hist_monthly.columns.tolist()

    # Fixed CPs (basins.json) are excluded from disaggregation entirely and spliced back in
    # after generation -- see toolkit.utils.fixed_control_points.
    free_sites, fixed_sites = split_free_and_fixed(site_names, basin)
    hist_monthly_free = hist_monthly[free_sites]

    outflow_index = free_sites.index(gage_name)
    num_years = len(hist_monthly) // 12
    historical_annual = hist_monthly[gage_name].resample("YS").sum().to_numpy()

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
    synthetic_h5_path = synthetic_dataset_path(filter_name, basin_name)

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
            bias_correction=BIAS_CORRECTION,
            historical_annual=historical_annual,
        )

        if fixed_sites:
            synthetic_streamflow_dict['streamflow'] = splice_fixed_columns(
                synthetic_streamflow_dict['streamflow'],
                free_sites,
                hist_monthly,
                basin,
                site_names,
            )
            synthetic_streamflow_dict['streamflow_columns'] = site_names

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
    print(f"Settings: FORCE_RECOMPUTE={FORCE_RECOMPUTE}, LOG_TRANSFORM={LOG_TRANSFORM}, "
          f"N_ENSEMBLES={N_ENSEMBLES}, STENCIL_SOURCE={STENCIL_SOURCE!r}, "
          f"STENCIL_PERIODS={STENCIL_PERIODS}, BIAS_CORRECTION={BIAS_CORRECTION}")

    args = parse_filter_basin_args('Generate synthetic streamflow for specific filter-basin combinations')

    with open(basins_path, "r") as f:
        BASINS = json.load(f)

    with open(ensemble_filters_path, "r") as f:
        ENSEMBLE_CONFIG = json.load(f)

    filter_sets, basins = select_filter_sets_and_basins(BASINS, ENSEMBLE_CONFIG, args.filter, args.basin)

    for filter_set in filter_sets:
        filter_name = filter_set["name"]
        ensemble_filters = filter_set["filters"]
        
        print(f"Processing filter: {filter_name}")
        
        for basin_name, basin in basins.items():
            print(f"  Generating synthetic streamflow for basin: {basin_name}")
            generate_synthetic_streamflow(basin_name, basin, ensemble_filters, filter_name)

if __name__ == "__main__":
    main()

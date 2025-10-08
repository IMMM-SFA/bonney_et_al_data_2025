"""
This script loads trained HMM models and generates synthetic streamflow.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
import json

from toolkit.hmm.model import BayesianStreamflowHMM
from toolkit.data.ninetyfiveofive import load_historical_data
from toolkit import repo_data_path, outputs_path
from toolkit.wrap.io import flo_to_df
from toolkit.data.io import save_netcdf_format, load_netcdf_format


### Settings ###
FORCE_RECOMPUTE = False # Whether to recompute the synthetic streamflow if it already exists
LOG_TRANSFORM = True # Whether to log transform the data
N_ENSEMBLES = 1000 # Number of ensembles to generate

### Path Configuration ###
basins_path = repo_data_path / "configs" / "basins.json"
ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters_basic.json"
# ensemble_filters_path = repo_data_path / "configs" / "ensemble_filters.json"

output_dir = outputs_path / "bayesian_hmm"

### Functions ###
def generate_synthetic_streamflow(basin_name, basin, ensemble_filters, filter_name):
    """Generate synthetic streamflow for a single basin using a trained HMM model."""
    if basin_name == "Rio Grande":
        return
    
    gage_name = basin["gage_name"]
    reach_id = basin["reach_id"]
    
    # Flo file
    flo_file = repo_data_path / basin["flo_file"]
    
    # Model path
    model_path = output_dir / "models" / f"{gage_name}_{filter_name}_model"
    
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

    # Load trained model
    model = BayesianStreamflowHMM.load(str(model_path))

    # Generate synthetic streamflow
    synthetic_h5_path = output_dir / f"{basin_name.lower().replace(' ', '_')}_{filter_name}_synthetic_streamflow.nc"

    if os.path.exists(synthetic_h5_path) and not FORCE_RECOMPUTE:
        synthetic_streamflow_dict = load_netcdf_format(synthetic_h5_path)
    else:
        synthetic_streamflow_dict = model.generate_synthetic_streamflow(
            start_year=2020,
            historical_monthly_data=hist_monthly.values,
            drought=None,
            random_seed=42,
            site_names=hist_monthly.columns.tolist(),
            time_index=hist_monthly.index.tolist(),
            h5_path=synthetic_h5_path,
            n_ensembles=N_ENSEMBLES
        )
        
        # Save synthetic streamflow to netcdf
        # Convert ensemble_filters to NetCDF-compatible format (remove None values)
        netcdf_filters = {k: v for k, v in ensemble_filters.items() if v is not None}
        
        global_metadata = {
            'basin_name': basin_name,
            'gage_name': gage_name,
            'reach_id': reach_id,
            'filter_name': filter_name,
            'ensemble_filters': str(netcdf_filters)  # Convert to string for NetCDF compatibility
        }
        save_netcdf_format(synthetic_streamflow_dict, synthetic_h5_path, additional_metadata=global_metadata)

### Main ###

# Load basin configuration from JSON
with open(basins_path, "r") as f:
    BASINS = json.load(f)

# Load ensemble filters configuration from JSON
with open(ensemble_filters_path, "r") as f:
    ENSEMBLE_CONFIG = json.load(f)

for filter_set in ENSEMBLE_CONFIG["filter_sets"]:
    filter_name = filter_set["name"]
    ensemble_filters = filter_set["filters"]
    
    # Generate synthetic streamflow for each basin with the current filter set
    for basin_name, basin in BASINS.items():
        generate_synthetic_streamflow(basin_name, basin, ensemble_filters, filter_name)

import os
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import logging
from sklearn.preprocessing import StandardScaler
from toolkit.data.metadata import filter_ensemble_members, create_metadata_df
import toolkit

logger = logging.getLogger(__name__)


def load_historical_data(
    flo_file: Path, 
    gage_name: str, 
    reach_id: int, 
    aggregate_annually: bool = True,
    log1p_transform: bool = False,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Load historical streamflow data from FLO file.
   
    Parameters
    ----------
    flo_file : Path
        Path to FLO file
    gage_name : str
        Name of the gage
    reach_id : int
        Reach ID for the gage
    aggregate_annually : bool, default=True
        Whether to aggregate to annual values
    log1p_transform : bool, default=False
        Whether to apply log1p transformation
    start_year : Optional[int], default=None
        Start year for filtering data (inclusive)
    end_year : Optional[int], default=None
        End year for filtering data (inclusive)
   
    Returns
    -------
    Tuple[pd.Series, Dict[str, Any]]
        Historical streamflow data and metadata
    """
    logger.info(f"Loading historical data for {gage_name} from {flo_file}")
    
    # Load C3.FLO data
    from toolkit.wrap.io import flo_to_df
    df = flo_to_df(flo_file)
    
    if gage_name not in df.columns:
        raise ValueError(f"Gage {gage_name} not found in C3.FLO file")
    
    # Get streamflow data
    hist_data = df[gage_name].copy()
    
    # Apply year filtering if specified
    if start_year is not None or end_year is not None:
        logger.info(f"Filtering historical data: start_year={start_year}, end_year={end_year}")
        if start_year is not None:
            hist_data = hist_data[hist_data.index.year >= start_year]
        if end_year is not None:
            hist_data = hist_data[hist_data.index.year <= end_year]
    
    # Aggregate to annual values if requested
    if aggregate_annually:
        logger.info("Aggregating historical data to annual values...")
        hist_data = hist_data.resample('YE').sum()
    
    # Apply log1p transformation if requested
    if log1p_transform:
        hist_data[:] = np.log1p(hist_data)
        logger.info("Applied log1p transformation to historical data")
    
    # Create metadata
    hist_metadata = {
        "gage_name": gage_name,
        "reach_id": reach_id,
        "n_timesteps": len(hist_data),
        "time_unit": "years" if aggregate_annually else "months",
        "time_range": {
            "start": str(hist_data.index[0]),
            "end": str(hist_data.index[-1])
        },
        "log1p_transformed": log1p_transform,
        "source_file": str(flo_file),
        "year_filtering": {
            "start_year": start_year,
            "end_year": end_year
        }
    }
    
    return hist_data, hist_metadata

def load_doe_data(
    nc_file: Path,
    flo_file: Path,
    gage_name: str,
    reach_id: int,
    period: str = "2020_2059",
    aggregate_annually: bool = True,
    log1p_transform: bool = True,
    ensemble_filters: Optional[Dict[str, Any]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load DOE (Department of Energy) ensemble streamflow data from NetCDF file.
   
    Parameters
    ----------
    nc_file : Path
        Path to NetCDF file with future streamflow data
    flo_file : Path
        Path to FLO file with historical streamflow data (for metadata)
    gage_name : str
        Name of the gage
    reach_id : int
        Reach ID for the gage
    period : str, default="2020_2059"
        Time period for future data
    aggregate_annually : bool, default=True
        Whether to aggregate to annual values
    log1p_transform : bool, default=True
        Whether to apply log1p transformation
    ensemble_filters : Optional[Dict[str, Any]], default=None
        Ensemble filters
    start_year : Optional[int], default=None
        Start year for filtering data (inclusive)
    end_year : Optional[int], default=None
        End year for filtering data (inclusive)
        
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        DOE data and metadata
    """
    logger.info(f"Loading streamflow data for {gage_name} (reach {reach_id}) with filtering")
    
    # Load NetCDF dataset
    with xr.open_dataset(nc_file) as ds:
        # Apply ensemble filtering if specified
        if ensemble_filters:
            logger.info(f"Applying filters: {ensemble_filters}")

            ds = filter_ensemble_members(ds, ensemble_filters)
        
        # Extract data for the specific reach
        var_name = f"reach_{reach_id}"
        if var_name not in ds:
            raise ValueError(f"Variable {var_name} not found in dataset")
        
        # Get data for filtered ensemble members
        doe_data = ds[var_name].values  # Shape: (n_ensembles, n_timesteps)
        
        # Get time information for year filtering
        time_values = ds.time_mn.values if 'time_mn' in ds else ds.time.values
        
    # Apply year filtering if specified
    if start_year is not None or end_year is not None:
        logger.info(f"Filtering DOE data: start_year={start_year}, end_year={end_year}")
        
        # Convert time values to years (assuming monthly data)
        if len(time_values) > 0:
            # Handle different time formats
            if hasattr(time_values[0], 'year'):
                # Already datetime objects
                years = np.array([t.year for t in time_values])
            else:
                # Assume numeric years or year-month format
                years = np.array([int(str(t)[:4]) for t in time_values])
            
            # Create mask for year filtering
            year_mask = np.ones(len(years), dtype=bool)
            if start_year is not None:
                year_mask &= (years >= start_year)
            if end_year is not None:
                year_mask &= (years <= end_year)
            
            # Apply mask to data
            doe_data = doe_data[:, year_mask]
            time_values = time_values[year_mask]
    
    # Create metadata dictionary
    doe_metadata = {
        "gage_name": gage_name,
        "reach_id": reach_id,
        "period": period,
        "n_ensembles": doe_data.shape[0],
        "n_timesteps": doe_data.shape[1],
        "time_range": {
            "start": str(time_values[0]) if len(time_values) > 0 else "Unknown",
            "end": str(time_values[-1]) if len(time_values) > 0 else "Unknown"
        },
        "ensemble_filters": ensemble_filters,
        "year_filtering": {
            "start_year": start_year,
            "end_year": end_year
        }
    }
    
    # Note: create_metadata_df requires the full dataset, so we'll skip it if we've filtered
    # doe_metadata = create_metadata_df(ds)
    
    # Aggregate future data to annual values if requested
    if aggregate_annually:
        logger.info("Aggregating future data to annual values...")
        if len(doe_data.shape) == 3:  # (n_members, n_years, 12)
            doe_data = np.sum(doe_data, axis=2)
        elif len(doe_data.shape) == 2:  # (n_members, n_years*12)
            # Reshape and sum
            n_members, n_months = doe_data.shape
            n_years = n_months // 12
            doe_data = doe_data.reshape(n_members, n_years, 12)
            doe_data = np.sum(doe_data, axis=2)
    
    # Apply log1p transformation to future data if requested
    if log1p_transform:
        logger.info("Applying log1p transformation to future data...")
        doe_data = np.log1p(doe_data)
    
    return doe_data, doe_metadata 
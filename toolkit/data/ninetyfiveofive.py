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


def _map_sites_to_reach_vars(site_names: List[str], pcp_reach_mapping: pd.DataFrame) -> List[str]:
    """Map FLO control-point names to their 9505 `reach_<COMID>` variable names.

    Whitespace-insensitive: FLO columns pad site codes to a fixed width (e.g. Trinity's
    "IN 8BEMA", Sabine's "IN  BARP"), while `pcp_to_reach_mapping.csv`'s PCP_NAME column
    does not, so both sides are compared with whitespace stripped.

    Parameters
    ----------
    site_names : List[str]
        Basin FLO column names, in the order the caller wants the output columns.
    pcp_reach_mapping : pd.DataFrame
        `outputs/9505/pcp_to_reach_mapping.csv`, with PCP_NAME and REACH_COMID columns.

    Returns
    -------
    List[str]
        `reach_<COMID>` variable name for each entry in `site_names`, same order.
    """
    stripped_to_comid = {
        str(name).replace(" ", ""): comid
        for name, comid in zip(pcp_reach_mapping["PCP_NAME"], pcp_reach_mapping["REACH_COMID"])
    }
    reach_vars = []
    for site in site_names:
        stripped = str(site).replace(" ", "")
        if stripped not in stripped_to_comid:
            raise ValueError(f"No 9505 reach mapping found for site {site!r}")
        reach_vars.append(f"reach_{stripped_to_comid[stripped]}")
    return reach_vars


def load_9505_stencil_pool(
    site_names: List[str],
    pcp_reach_mapping: pd.DataFrame,
    nc_paths: Dict[str, Path],
    periods: List[str],
    ensemble_filters: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Build a monthly candidate "stencil" pool from the DOE 9505 ensemble, for use as
    `historical_monthly_data` in `toolkit.hmm.disaggregation.disaggregate_annual_to_monthly`.

    Each 9505 (ensemble member, year) pair becomes one 12-month candidate block, so the pool
    can be far larger than the single observed historical record `disaggregate_annual_to_monthly`
    otherwise draws from -- but the block layout is identical (`(n_blocks*12, n_sites)`), since
    that function already treats its input as an arbitrary stack of 12-month blocks rather than
    one contiguous historical series (it fabricates its own sequential year index internally).
    To blend with the observed historical record, concatenate this function's output with the
    historical FLO array along axis 0 before passing it on.

    Parameters
    ----------
    site_names : List[str]
        Basin FLO column names (e.g. `flo_to_df(...).columns.tolist()`), in the exact order the
        output columns must match so callers' `anchor_index`/`outflow_index` stay valid.
    pcp_reach_mapping : pd.DataFrame
        `outputs/9505/pcp_to_reach_mapping.csv`, with PCP_NAME and REACH_COMID columns.
    nc_paths : Dict[str, Path]
        Period name -> path to that period's `master_streamflow_{period}_af.nc` (already in
        acre-feet/month, matching `.FLO` file units).
    periods : List[str]
        Which period(s) (keys into `nc_paths`) to pool candidate blocks from.
    ensemble_filters : Optional[Dict[str, Any]], default=None
        Same filter dict used to train the basin's HMM (`toolkit.data.metadata.filter_ensemble_members`),
        so stencils are drawn from the same ensemble-member subset the annual model was trained on.

    Returns
    -------
    np.ndarray
        Shape (n_candidate_blocks * 12, n_sites), columns ordered to match `site_names`.
    """
    reach_vars = _map_sites_to_reach_vars(site_names, pcp_reach_mapping)

    period_blocks = []
    for period in periods:
        with xr.open_dataset(nc_paths[period]) as ds:
            if ensemble_filters:
                ds = filter_ensemble_members(ds, ensemble_filters)
            # (n_members, n_time, n_sites), one column per site in site_names order
            site_arrays = [ds[var].values for var in reach_vars]
            period_data = np.stack(site_arrays, axis=-1)
        # Flatten (n_members, n_time) -> n_members * n_time rows; each site's 12-month blocks
        # stay contiguous within a member since n_time is already a whole number of years.
        period_blocks.append(period_data.reshape(-1, period_data.shape[-1]))

    return np.concatenate(period_blocks, axis=0)

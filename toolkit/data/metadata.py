import pandas as pd
import xarray as xr
from typing import Union, List, Optional, Dict, Any

def create_metadata_df(ds: xr.Dataset) -> pd.DataFrame:
    """
    Create a metadata DataFrame by parsing filenames from ds['original_filename'].
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing original_filename array
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing metadata for each ensemble member
        
    Notes
    -----
    Filename format: {hydro_model}_RAPID_{gcm}_{scenario}_{variant}_{downscaling}_{rmf}_{period}.nc
    Example: PRMS_RAPID_BCC-CSM2-MR_ssp245_r1i1p1f1_DBCCA_Daymet_2020_2059.nc
    """
    import re
    
    # Get original filenames and ensemble_ids
    filenames = ds['original_filename'].values
    ensemble_ids = ds['ensemble_id'].values
    
    # Initialize lists for metadata
    hydro_models = []
    gcms = []
    scenarios = []
    variants = []
    downscaling_methods = []
    rmf_methods = []
    periods = []
    
    # Parse each filename
    for filename in filenames:
        # Remove .nc extension
        name = filename.replace('.nc', '')
        
        # Split by underscore
        parts = name.split('_')
        
        if len(parts) >= 8:
            # Expected format: {hydro_model}_RAPID_{gcm}_{scenario}_{variant}_{downscaling}_{rmf}_{period}
            hydro_model = parts[0]  # PRMS, VIC5, etc.
            gcm = parts[2]  # BCC-CSM2-MR, ACCESS-CM2, etc.
            scenario = parts[3]  # ssp245, ssp585, etc.
            variant = parts[4]  # r1i1p1f1, r1i1p1f2, etc.
            downscaling = parts[5]  # DBCCA, RegCM, etc.
            rmf = parts[6]  # Daymet, Livneh, etc.
            period = '_'.join(parts[7:])  # 2020_2059, etc.
            
            hydro_models.append(hydro_model)
            gcms.append(gcm)
            scenarios.append(scenario)
            variants.append(variant)
            downscaling_methods.append(downscaling)
            rmf_methods.append(rmf)
            periods.append(period)
        else:
            # Fallback for unexpected format
            hydro_models.append('Unknown')
            gcms.append('Unknown')
            scenarios.append('Unknown')
            variants.append('Unknown')
            downscaling_methods.append('Unknown')
            rmf_methods.append('Unknown')
            periods.append('Unknown')
    
    # Create DataFrame
    meta_df = pd.DataFrame({
        'ensemble_id': ensemble_ids,
        'hydro_model': hydro_models,
        'gcm': gcms,
        'scenario': scenarios,
        'variant': variants,
        'downscaling': downscaling_methods,
        'rmf': rmf_methods,
        'period': periods,
        'original_filename': filenames
    })
    
    # Add legacy column names for backward compatibility
    meta_df['prefix1'] = meta_df['hydro_model']  # For backward compatibility
    meta_df['bias_correction'] = meta_df['rmf']  # rmf serves as bias correction method
    
    return meta_df

def filter_ensemble_members(ds: xr.Dataset, ensemble_filters: Dict[str, Any]) -> xr.Dataset:
    """
    Filter dataset to specific ensemble members based on metadata criteria.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset to filter
    ensemble_filters : Dict[str, Any]
        Dictionary containing filter criteria. Supported keys:
        - 'rmf': Regional meteorological forcing
        - 'hydro_model': Hydrological model  
        - 'downscaling': Downscaling method
        - 'scenarios': List of climate scenarios
        - 'bias_correction': Bias correction method
        
    Returns
    -------
    xarray.Dataset
        Filtered dataset containing only the desired ensemble members
        
    Raises
    ------
    ValueError
        If no ensemble members match the given criteria
    """
    # Get ensemble metadata
    ensemble_meta = create_metadata_df(ds)
    
    # Extract filter values from dictionary
    rmf = ensemble_filters.get("rmf")
    hydro_model = ensemble_filters.get("hydro_model")
    downscaling = ensemble_filters.get("downscaling")
    scenarios = ensemble_filters.get("scenarios")
    bias_correction = ensemble_filters.get("bias_correction")
    gcm = ensemble_filters.get("gcm")
    
    # Build filter conditions
    conditions = []
    if rmf is not None:
        conditions.append(ensemble_meta["rmf"].isin(rmf))
    if hydro_model is not None:
        conditions.append(ensemble_meta["prefix1"].isin(hydro_model))
    if downscaling is not None:
        conditions.append(ensemble_meta["downscaling"].isin(downscaling))
    if scenarios is not None:
        conditions.append(ensemble_meta["scenario"].isin(scenarios))
    if bias_correction is not None:
        conditions.append(ensemble_meta["bias_correction"].isin(bias_correction))
    if gcm is not None:
        conditions.append(ensemble_meta["gcm"].isin(gcm))
    
    # Apply filters
    if conditions:
        mask = conditions[0]
        for cond in conditions[1:]:
            mask = mask & cond
        valid_ensembles = ensemble_meta[mask]["ensemble_id"].unique()
    else:
        valid_ensembles = ensemble_meta["ensemble_id"].unique()
    
    if len(valid_ensembles) == 0:
        raise ValueError("No ensemble members matched the given criteria.")
    
    # Subset the dataset
    return ds.sel(ensemble_id=valid_ensembles)
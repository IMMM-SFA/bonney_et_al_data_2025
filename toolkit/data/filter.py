import numpy as np
from pathlib import Path
import re
import pandas as pd
import xarray as xr
from typing import Union, Optional
import geopandas as gpd
from typing import Dict, List
import logging

def create_basin_subsets(
    huc8_gdf: gpd.GeoDataFrame, 
    reaches_gdf: gpd.GeoDataFrame, 
    basins_gdf: gpd.GeoDataFrame,
    basin_names: List[str]) -> Dict[str, Dict[str, gpd.GeoDataFrame]]:
    """Create subsets of HUC8 and reach geometries for specified basins."""
    basin_subsets = {}
    
    for basin_name in basin_names:
        logging.info(f"Processing basin: {basin_name}")
        
        # Get the basin geometry
        basin = basins_gdf[basins_gdf['basin_name'] == basin_name].iloc[0]
        
        # Create subsets
        basin_huc8 = huc8_gdf[huc8_gdf.intersects(basin.geometry)].copy()
        basin_reaches = reaches_gdf[reaches_gdf.intersects(basin.geometry)].copy()
        
        basin_subsets[basin_name] = {
            'huc8': basin_huc8,
            'reaches': basin_reaches
        }
        
        logging.info(f"Found {len(basin_huc8)} HUC8s and {len(basin_reaches)} reaches in {basin_name}")
    
    return basin_subsets

def find_connecting_reaches(basin_subsets: Dict[str, Dict[str, gpd.GeoDataFrame]]) -> Dict[str, gpd.GeoDataFrame]:
    """Find reaches that connect pairs of HUC8s within each basin."""
    connecting_reaches = {}
    
    for basin_name, subsets in basin_subsets.items():
        logging.info(f"Finding connecting reaches for {basin_name}")
        
        huc8_gdf = subsets['huc8']
        reaches_gdf = subsets['reaches']
        
        # Create a new GeoDataFrame for connecting reaches
        connecting_reaches_gdf = gpd.GeoDataFrame(columns=['geometry', 'huc8_pair'], crs=reaches_gdf.crs)
        
        # Iterate through all pairs of HUC8s
        for i, huc8_1 in huc8_gdf.iterrows():
            for j, huc8_2 in huc8_gdf.iterrows():
                if i >= j:  # Skip self and already checked pairs
                    continue
                
                # Find reaches that intersect both HUC8s
                connecting = reaches_gdf[
                    reaches_gdf.intersects(huc8_1.geometry) & 
                    reaches_gdf.intersects(huc8_2.geometry)
                ].copy()
                
                if not connecting.empty:
                    # Add the HUC8 pair information
                    connecting['huc8_pair'] = f"{huc8_1['HUC08']}_{huc8_2['HUC08']}"
                    connecting_reaches_gdf = pd.concat([connecting_reaches_gdf, connecting])
        
        connecting_reaches[basin_name] = connecting_reaches_gdf
        logging.info(f"Found {len(connecting_reaches_gdf)} connecting reaches in {basin_name}")
    
    return connecting_reaches

def find_nearest_gage(reaches_gdf: gpd.GeoDataFrame, gages_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Find the nearest gage to each reach."""
    logging.info("Finding nearest gages to reaches...")
    
    # Create a new GeoDataFrame to store results
    results = reaches_gdf.copy()
    results['associated_gage_idx'] = None
    results['distance_to_gage_m'] = np.nan
    
    # Dictionary to store the nearest gage for each reach
    reach_to_gage = {}
    
    for reach_idx, reach in reaches_gdf.iterrows():
        try:
            # Create a buffer around the reach (in meters since we're in UTM)
            buffer_size = 1000  # 1km buffer
            reach_buffer = reach.geometry.buffer(buffer_size)
            
            # Find gages that intersect with the buffer
            intersecting_gages = gages_gdf[gages_gdf.intersects(reach_buffer)]
            
            if len(intersecting_gages) > 0:
                # Calculate distances to all intersecting gages
                distances = intersecting_gages.geometry.distance(reach.geometry)
                
                # Find the nearest gage
                nearest_gage_idx = distances.idxmin()
                min_distance = distances.min()
                
                # Store the association
                reach_to_gage[reach_idx] = {
                    'gage_idx': nearest_gage_idx,
                    'distance': float(min_distance)
                }
                
                if reach_idx % 100 == 0:
                    logging.info(f"Processed {reach_idx} reaches")
            else:
                logging.warning(f"No gages found within {buffer_size}m of reach {reach_idx}")
                
        except Exception as e:
            logging.warning(f"Error processing reach {reach_idx}: {str(e)}")
            continue
    
    # Update the results GeoDataFrame with gage associations
    for reach_idx, info in reach_to_gage.items():
        results.at[reach_idx, 'associated_gage_idx'] = info['gage_idx']
        results.at[reach_idx, 'distance_to_gage_m'] = info['distance']
    
    # Convert distance column to numeric, replacing any non-numeric values with NaN
    results['distance_to_gage_m'] = pd.to_numeric(results['distance_to_gage_m'], errors='coerce')
    
    # Filter to only include reaches that have associated gages
    results = results[results['associated_gage_idx'].notna()]
    
    logging.info(f"Found {len(results)} reaches with associated gages")
    
    return results


def fetch_filtered_files(folder_path, model=None, scenario=None, variant_id=None, downscaling=None, rmf=None, year_start=None, year_end=None):
    """
    Fetches NetCDF file paths based on filtering criteria using simple text search.

    Parameters:
        folder_path (str or Path): Path to the directory containing NetCDF files.
        model, scenario, variant_id, downscaling, rmf, year_start, year_end (str, optional): Filtering keywords.
    
    Returns:
        List of Path objects: Filtered file paths matching the criteria.
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise ValueError(f"Folder path {folder_path} does not exist.")

    # Convert filter parameters into a list of keywords
    keywords = [model, scenario, variant_id, downscaling, rmf, year_start, year_end]
    keywords = [kw for kw in keywords if kw]  # Remove None values

    # Find all NetCDF files
    all_files = list(folder_path.glob("*.nc"))

    # Filter files based on keyword presence in filename
    filtered_files = [f for f in all_files if all(kw in f.name for kw in keywords)]

    return filtered_files


def filter_dataset(streamflow_dict, shortage_dict, train_datadict):
    """Takes streamflow and shortage data and filters the columns
    based on out the data in train_datadict were filtered

    Parameters
    ----------
    streamflow_dict : dict
        _description_
    shortage_dict : dict
        _description_
    train_datadict : dict
        _description_

    Returns
    -------
    dict
        _description_
    """
    
    streamflow_data = streamflow_dict["streamflow_data"]
    streamflow_index = streamflow_dict["streamflow_index"]
    streamflow_columns = streamflow_dict["streamflow_columns"]

    shortage_data = shortage_dict["shortage_data"]
    shortage_index = shortage_dict["shortage_index"]
    shortage_columns = shortage_dict["shortage_columns"]

    # 
    train_shortage_index = train_datadict["shortage_index"]
    index_mask = np.isin(shortage_index, train_shortage_index)
    assert (shortage_index[index_mask] == train_shortage_index).all()

    # 
    train_shortage_columns = train_datadict["shortage_columns"]
    column_mask = np.isin(shortage_columns, train_shortage_columns)
    assert (shortage_columns[column_mask] == train_shortage_columns).all()

    # 
    shortage_data = shortage_data[:,index_mask,:]
    shortage_data = shortage_data[:,:,column_mask]
    shortage_index = shortage_index[index_mask]
    shortage_columns = shortage_columns[column_mask]

    # 
    streamflow_data = streamflow_data[:,index_mask,:]
    streamflow_index = streamflow_index[index_mask]

    right_sectors = train_datadict["sector"]
    right_seniority = train_datadict["seniority"]
    right_allotments = train_datadict["allotment"]

    # converting nans
    shortage_data[np.isnan(shortage_data)] = 0.0

    # assert that data structures shapes line up as a final check
    assert streamflow_data.shape[1] == streamflow_index.shape[0]
    assert streamflow_data.shape[2] == streamflow_columns.shape[0]

    assert shortage_data.shape[1] == shortage_index.shape[0]
    assert shortage_data.shape[2] == shortage_columns.shape[0]
    assert shortage_data.shape[2] == right_sectors.shape[0]
    assert shortage_data.shape[2] == right_seniority.shape[0]

    assert shortage_data.shape[0] == streamflow_data.shape[0]
    assert shortage_data.shape[1] == streamflow_data.shape[1]

    assert (streamflow_index == shortage_index).all()


    # organize data back into a dictionary
    data_dict = dict() 
    data_dict["streamflow_data"] = streamflow_data
    data_dict["streamflow_index"] = streamflow_index
    data_dict["streamflow_columns"] = streamflow_columns

    data_dict["shortage_data"] = shortage_data
    data_dict["shortage_index"] = shortage_index
    data_dict["shortage_columns"] = shortage_columns
    data_dict["sector"] = right_sectors
    data_dict["seniority"] = right_seniority
    data_dict["allotment"] = right_allotments

    print("Final streamflow data shape:", streamflow_data.shape)
    print("Final shortage data shape:", shortage_data.shape)

    return data_dict


def subset_data_by_ensemble(dataset: Union[pd.DataFrame, xr.Dataset], 
                          downscaling_type: Optional[str] = None, 
                          model: Optional[str] = None,
                          scenario: Optional[str] = None) -> Union[pd.DataFrame, xr.Dataset]:
    """
    Create a subset of the data based on filtering rules for the ensemble.
    
    Parameters
    ----------
    dataset : DataFrame or Dataset
        The dataset to filter
    downscaling_type : str, optional
        Type of downscaling to filter by
    model : str, optional
        Model name to filter by
    scenario : str, optional
        Scenario name to filter by
        
    Returns
    -------
    DataFrame or Dataset
        Filtered dataset
    """
    if dataset is None:
        raise ValueError("No dataset provided")
    
    # Implementation depends on the dataset structure
    # For xarray Dataset
    if isinstance(dataset, xr.Dataset):
        subset = dataset
        
        # Apply filters if they exist as dimensions or coordinates
        for dim_name, filter_val in [('downscaling', downscaling_type),
                                    ('model', model),
                                    ('scenario', scenario)]:
            if filter_val and dim_name in subset.dims:
                subset = subset.sel({dim_name: filter_val})
            elif filter_val and dim_name in subset.coords:
                subset = subset.where(subset[dim_name] == filter_val, drop=True)
        
        return subset
    
    # For pandas DataFrame
    elif isinstance(dataset, pd.DataFrame):
        subset = dataset.copy()
        
        # Apply filters if the columns exist
        if downscaling_type and 'downscaling' in subset.columns:
            subset = subset[subset['downscaling'] == downscaling_type]
        
        if model and 'model' in subset.columns:
            subset = subset[subset['model'] == model]
        
        if scenario and 'scenario' in subset.columns:
            subset = subset[subset['scenario'] == scenario]
        
        return subset
    
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")
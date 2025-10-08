"""
This script finds the nearest reach to each control point and saves the results to a shapefile and CSV.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from shapely.ops import nearest_points
import numpy as np
from toolkit import outputs_path, repo_data_path
import os

### Settings ###
# None

### Path Configuration ###
reach_paths = [
    os.path.join(repo_data_path, "geospatial", "9505_shapefiles", "NHDFlowline12.shp"),
    os.path.join(repo_data_path, "geospatial", "9505_shapefiles", "NHDFlowline13.shp"),
    os.path.join(repo_data_path, "geospatial", "9505_shapefiles", "NHDFlowline11.shp"),
]
gage_paths = [
    os.path.join(repo_data_path, "geospatial", "wrap_gages", "Primary_CP_colorado.shp"),
    os.path.join(repo_data_path, "geospatial", "wrap_gages", "Primary_CP_sabine.shp"),
    os.path.join(repo_data_path, "geospatial", "wrap_gages", "Primary_CP_trinity.shp"),
]

output_dir = outputs_path / "9505"
output_dir.mkdir(exist_ok=True)

reach_shp_path = output_dir / "reaches_with_associated_gages.shp"

pcp_reach_path = output_dir / "pcp_to_reach_mapping.csv"

### Functions ###

def load_shapefiles(reach_paths: List[str], gage_paths: List[str]) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load the reach shapefiles and gage locations shapefiles.
    
    Parameters
    ----------
    reach_paths : List[str]
        List of paths to river reaches shapefiles
    gage_paths : List[str]
        List of paths to USGS gage locations shapefiles
        
    Returns
    -------
    Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        Reaches and gages GeoDataFrames
    """
    
    # Load and concatenate all gage shapefiles
    gage_gdfs = []
    for path in gage_paths:
        gage_gdf = gpd.read_file(path)
        gage_gdfs.append(gage_gdf)
    
    # Concatenate all gage GeoDataFrames
    gages_gdf = pd.concat(gage_gdfs, ignore_index=True)
    
    # Load and concatenate all reach shapefiles
    reach_gdfs = []
    for path in reach_paths:
        reach_gdf = gpd.read_file(path)
        reach_gdfs.append(reach_gdf)
    
    # Concatenate all reach GeoDataFrames
    reaches_gdf = pd.concat(reach_gdfs, ignore_index=True)
    
    # Ensure both GeoDataFrames are in the same CRS
    if not (reaches_gdf.crs == gages_gdf.crs):
        gages_gdf = gages_gdf.to_crs(reaches_gdf.crs)
    
    # Project to a suitable UTM zone for distance calculations
    # First, determine a suitable UTM zone based on the center of the data
    center = gages_gdf.geometry.unary_union.centroid
    utm_zone = int(np.floor((center.x + 180) / 6) + 1)
    utm_crs = f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs"
    
    reaches_gdf = reaches_gdf.to_crs(utm_crs)
    gages_gdf = gages_gdf.to_crs(utm_crs)
    
    return reaches_gdf, gages_gdf

def find_nearest_reach(gages_gdf: gpd.GeoDataFrame, reaches_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Find the nearest reach to each gage location and return a GeoDataFrame of reaches
    with their associated gage information.
    
    Parameters
    ----------
    gages_gdf : gpd.GeoDataFrame
        GeoDataFrame containing gage locations
    reaches_gdf : gpd.GeoDataFrame
        GeoDataFrame containing river reaches
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame of reaches with associated gage information
    """
    # Create a new GeoDataFrame to store results
    results = reaches_gdf.copy()
    results['associated_gage_idx'] = None
    results['distance_to_gage_m'] = np.nan
    
    # Dictionary to store the nearest reach for each gage
    gage_to_reach = {}
    
    for gage_idx, gage in gages_gdf.iterrows():
        try:
            # Create a buffer around the gage (in meters since we're in UTM)
            buffer_size = 1000  # 1km buffer
            gage_buffer = gage.geometry.buffer(buffer_size)
            
            # Find reaches that intersect with the buffer
            intersecting_reaches = reaches_gdf[reaches_gdf.intersects(gage_buffer)]
            
            if len(intersecting_reaches) > 0:
                # Calculate distances to all intersecting reaches
                distances = intersecting_reaches.geometry.distance(gage.geometry)
                
                # Find the nearest reach
                nearest_reach_idx = distances.idxmin()
                min_distance = distances.min()
                
                # Store the association
                gage_to_reach[nearest_reach_idx] = {
                    'gage_idx': gage_idx,
                    'distance': float(min_distance)
                }
                
            else:
                pass
                
        except Exception as e:
            continue
    
    # Update the results GeoDataFrame with gage associations
    for reach_idx, info in gage_to_reach.items():
        results.at[reach_idx, 'associated_gage_idx'] = info['gage_idx']
        results.at[reach_idx, 'distance_to_gage_m'] = info['distance']
    
    # Convert distance column to numeric, replacing any non-numeric values with NaN
    results['distance_to_gage_m'] = pd.to_numeric(results['distance_to_gage_m'], errors='coerce')
    
    # Filter to only include reaches that have associated gages
    results = results[results['associated_gage_idx'].notna()]
    
    return results

### Main ###

# Load shapefiles
reaches_gdf, gages_gdf = load_shapefiles(reach_paths, gage_paths)

# Find nearest reaches
results_gdf = find_nearest_reach(gages_gdf, reaches_gdf)

# Save results
results_gdf.to_file(reach_shp_path)

# Create combined dataframe of PCPs and their reaches
pcp_reach_df = pd.DataFrame({
    'PCP_NAME': gages_gdf.loc[results_gdf['associated_gage_idx'], 'ID'].values,
    'REACH_COMID': results_gdf['COMID'].values,
    'DISTANCE_TO_GAGE_M': results_gdf['distance_to_gage_m'].values,
    # 'BASIN': gages_gdf.loc[results_gdf['associated_gage_idx'], 'BASIN'].values
})

# Sort by basin and PCP name
pcp_reach_df = pcp_reach_df.sort_values(['PCP_NAME'])

# Save to CSV
pcp_reach_df.to_csv(pcp_reach_path, index=False)

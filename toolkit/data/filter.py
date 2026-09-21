import geopandas as gpd
import pandas as pd
import numpy as np
from typing import List, Set, Tuple


def load_gage_and_reach_shapefiles(reach_paths: List[str], gage_paths: List[str]) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load the reach shapefiles and gage locations shapefiles, reprojected to a shared
    UTM zone suitable for distance calculations.

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
    gage_gdfs = [gpd.read_file(path) for path in gage_paths]
    gages_gdf = pd.concat(gage_gdfs, ignore_index=True)

    # Load and concatenate all reach shapefiles
    reach_gdfs = [gpd.read_file(path) for path in reach_paths]
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


def associate_gages_to_reaches(gages_gdf: gpd.GeoDataFrame, reaches_gdf: gpd.GeoDataFrame, buffer_size: float = 2000) -> gpd.GeoDataFrame:
    """
    Find the nearest reach to each gage location and return a GeoDataFrame of reaches
    with their associated gage information.

    Parameters
    ----------
    gages_gdf : gpd.GeoDataFrame
        GeoDataFrame containing gage locations
    reaches_gdf : gpd.GeoDataFrame
        GeoDataFrame containing river reaches
    buffer_size : float, default=2000
        Search radius (in the CRS's units, expected to be meters) around each gage
        used to find candidate reaches. 1km missed Trinity's IN8CEMA by ~170m.

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

        except Exception as e:
            print(f"Warning: Could not associate gage {gage_idx}: {e}")
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


def find_huc8_codes_for_reaches(reach_paths: List[str], comids: List[int]) -> Set[str]:
    """
    Look up the HUC8 subregion covering each of the given reach COMIDs, by matching
    against the COMID/HUC08 attributes of the given NHDFlowline shapefiles. Used to
    scope the 9505 download to only the HUC8s that actually contain reaches of interest,
    instead of every HUC8 under a broader HUC2 region.

    Parameters
    ----------
    reach_paths : List[str]
        Paths to NHDFlowline shapefiles (each must have COMID and HUC08 fields).
    comids : List[int]
        Reach COMIDs to look up.

    Returns
    -------
    Set[str]
        Zero-padded 8-digit HUC8 codes covering the given COMIDs.
    """
    remaining = set(int(c) for c in comids)
    huc8_codes = set()

    for path in reach_paths:
        gdf = gpd.read_file(path, columns=['COMID', 'HUC08'], ignore_geometry=True)
        matches = gdf[gdf['COMID'].isin(remaining)]
        huc8_codes.update(str(int(code)).zfill(8) for code in matches['HUC08'])
        remaining -= set(matches['COMID'])

    if remaining:
        raise ValueError(f"No HUC8 found for COMIDs (not present in any reach_paths shapefile): {sorted(remaining)}")

    return huc8_codes

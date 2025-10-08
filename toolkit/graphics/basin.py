import matplotlib.pyplot as plt
import geopandas as gpd
from typing import Optional, Dict
import pandas as pd
import xarray as xr


def plot_basin_boundaries(basin_shapes: gpd.GeoDataFrame, 
                         highlight_basin: Optional[str] = None,
                         get_basin_shape_func=None) -> plt.Figure:
    """
    Plot the boundaries of all basins, optionally highlighting one.
    
    Parameters
    ----------
    basin_shapes : GeoDataFrame
        GeoDataFrame containing all basin shapes
    highlight_basin : str, optional
        Name of the basin to highlight
    get_basin_shape_func : callable, optional
        Function to get a specific basin shape by name
        
    Returns
    -------
    Figure
        Matplotlib figure with the plot
    """
    if basin_shapes is None:
        raise ValueError("Basin shapes not provided")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all basins
    basin_shapes.plot(ax=ax, color='lightgray', edgecolor='black')
    
    # Highlight the selected basin if specified
    if highlight_basin and get_basin_shape_func:
        basin_shape = get_basin_shape_func(highlight_basin)
        if basin_shape is not None and len(basin_shape) > 0:
            basin_shape.plot(ax=ax, color='red', edgecolor='black')
            ax.set_title(f"Texas River Basins (Highlighted: {highlight_basin})")
        else:
            ax.set_title("Texas River Basins")
    else:
        ax.set_title("Texas River Basins")
    
    return fig

def plot_huc8_shapes(huc8_shapes: gpd.GeoDataFrame, 
                    basin_name: Optional[str] = None, 
                    show_basin: bool = True,
                    basin_shapes: Optional[gpd.GeoDataFrame] = None,
                    get_basin_shape_func=None) -> plt.Figure:
    """
    Plot HUC8 watersheds, optionally for a specific basin.
    
    Parameters
    ----------
    huc8_shapes : GeoDataFrame
        GeoDataFrame containing HUC8 shapes
    basin_name : str, optional
        Name of the basin to filter HUC8s for
    show_basin : bool, default=True
        Whether to show the basin boundary
    basin_shapes : GeoDataFrame, optional
        GeoDataFrame containing basin shapes
    get_basin_shape_func : callable, optional
        Function to get a specific basin shape by name
        
    Returns
    -------
    Figure
        Matplotlib figure with the plot
    """
    if huc8_shapes is None:
        raise ValueError("HUC8 shapes not provided")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot HUC8s
    if basin_name:
        # Filter HUC8s for the specified basin
        huc8s = find_huc8s_in_basin(huc8_shapes, basin_name, get_basin_shape_func)
        if huc8s is not None:
            huc8s.plot(ax=ax, color='lightblue', edgecolor='black')
            title = f"HUC8 Watersheds in {basin_name} Basin"
        else:
            ax.set_title(f"No HUC8s found for {basin_name} basin")
            return fig
    else:
        # Plot all HUC8s
        huc8_shapes.plot(ax=ax, color='lightblue', edgecolor='black')
        title = "All HUC8 Watersheds"
    
    # Show basin boundary if requested
    if show_basin and basin_name and basin_shapes is not None and get_basin_shape_func:
        basin_shape = get_basin_shape_func(basin_name)
        if basin_shape is not None:
            basin_shape.plot(ax=ax, color='none', edgecolor='red', linewidth=2)
    
    ax.set_title(title)
    return fig


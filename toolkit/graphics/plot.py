import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from pathlib import Path
import matplotlib.colors as colors
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import List, Dict, Union, Optional, Tuple, Any
import calendar
import re
from datetime import datetime, timedelta
import matplotlib.dates as mdates


def plot_multi_period_seasonal_ribbon(
    datasets: List[xr.Dataset],
    labels: List[str],
    huc8_ids: List[str],
    variable: str = "streamflow_median",
    colors: Optional[List[str]] = None,
    ylabel: str = "Streamflow (CFS)",
    title: str = "Seasonal Streamflow Trends",
    show_std_dev: bool = True,
) -> plt.Figure:
    """
    Plot seasonal streamflow ribbon plots for multiple time periods.

    Parameters
    ----------
    datasets : List[xr.Dataset]
        List of xarray Datasets, one per time period.
    labels : List[str]
        Labels for each dataset (e.g., ["2020–2059", "2060–2099"]).
    huc8_ids : List[str]
        List of HUC8 watershed IDs to include.
    variable : str
        Variable to plot (e.g., "streamflow_median").
    colors : List[str], optional
        List of colors to use for each time period.
    ylabel : str
        Y-axis label.
    title : str
        Plot title.
    show_std_dev : bool
        Whether to show standard deviation shading.
    
    Returns
    -------
    plt.Figure
    """
    assert len(datasets) == len(labels), "Each dataset must have a label"
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(datasets)))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (ds, label) in enumerate(zip(datasets, labels)):
        # Subset to HUC8s of interest
        ds_huc = ds.sel(huc8=ds.huc8.isin(huc8_ids))
        
        # Extract month from time_mn (assuming time_mn starts at 0)
        months = (ds_huc["time_mn"] % 12) + 1  # Month 1–12
        ds_huc = ds_huc.assign_coords(month=("time_mn", months.data))

        # Take mean across huc8 dimension
        ds_huc_mean = ds_huc.mean(dim="huc8", skipna=True)

        # Group by month
        ts = ds_huc_mean[variable].groupby("month")

        median_vals = ts.median(dim=["ensemble_id", "time_mn"], skipna=True)
        q25_vals = ts.quantile(0.25, dim=["ensemble_id", "time_mn"], skipna=True)
        q75_vals = ts.quantile(0.75, dim=["ensemble_id", "time_mn"], skipna=True)

        # Update label to include the start and end years
        start_year = int(ds_huc["time_mn"].min().item() // 12 + 1)
        end_year = int(ds_huc["time_mn"].max().item() // 12 + 1)
        period_label = f"{start_year}-{end_year}"

        # Plot median and quartiles
        ax.plot(median_vals["month"], median_vals, label=f"{label} (Median)", color=colors[i], lw=2)
        ax.plot(q25_vals["month"], q25_vals, color=colors[i], linestyle="--", alpha=0.7, label=f"{label} (25th/75th Quartiles)")
        ax.plot(q75_vals["month"], q75_vals, color=colors[i], linestyle="--", alpha=0.7)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig


def plot_multi_period_annual_ribbon(
    datasets: List[xr.Dataset],
    labels: List[str],
    huc8_ids: List[str],
    variable: str = "streamflow_median",
    colors: Optional[List[str]] = None,
    ylabel: str = "Streamflow (CFS)",
    title: str = "Annual Streamflow Trends",
    show_std_dev: bool = True,
) -> plt.Figure:
    """
    Plot annual streamflow ribbon plots for multiple time periods.

    Parameters
    ----------
    datasets : List[xr.Dataset]
        List of xarray Datasets, one per time period.
    labels : List[str]
        Labels for each dataset (e.g., ["2020–2059", "2060–2099"]).
    huc8_ids : List[str]
        List of HUC8 watershed IDs to include.
    variable : str
        Variable to plot (e.g., "streamflow_median").
    colors : List[str], optional
        List of colors to use for each time period.
    ylabel : str
        Y-axis label.
    title : str
        Plot title.
    show_std_dev : bool
        Whether to show standard deviation shading.
    
    Returns
    -------
    plt.Figure
    """
    assert len(datasets) == len(labels), "Each dataset must have a label"
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(datasets)))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (ds, label) in enumerate(zip(datasets, labels)):
        # Subset to HUC8s of interest
        ds_huc = ds.sel(huc8=ds.huc8.isin(huc8_ids))
        
        # Take mean across huc8 dimension
        ds_huc_mean = ds_huc.mean(dim="huc8", skipna=True)

        # Group by year (every 12 months)
        years = ds_huc_mean["time_mn"] // 12 + 1  # Year 1, 2, ...
        ds_huc_mean = ds_huc_mean.assign_coords(year=("time_mn", years.data))
        ts = ds_huc_mean[variable].groupby("year")

        # Compute median and quartiles across ensemble_id for each year
        median_vals = ts.median(dim=["ensemble_id", "time_mn"], skipna=True)
        q25_vals = ts.quantile(0.25, dim=["ensemble_id", "time_mn"], skipna=True)
        q75_vals = ts.quantile(0.75, dim=["ensemble_id", "time_mn"], skipna=True)

        # Update label to include the start and end years
        start_year = int(ds_huc["time_mn"].min().item() // 12 + 1)
        end_year = int(ds_huc["time_mn"].max().item() // 12 + 1)
        period_label = f"{start_year}-{end_year}"

        # Plot median and quartiles
        ax.plot(median_vals["year"], median_vals, label=f"{label} (Median)", color=colors[i], lw=2)
        ax.plot(q25_vals["year"], q25_vals, color=colors[i], linestyle="--", alpha=0.7, label=f"{label} (25th/75th Quartiles)")
        ax.plot(q75_vals["year"], q75_vals, color=colors[i], linestyle="--", alpha=0.7)

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig


def plot_variables_over_time(dataset: xr.Dataset, ensemble_id_idx: Union[str, int], 
                             huc8_code: Optional[Union[str, int]] = None,
                             basin_name: Optional[str] = None,
                             variables: Optional[List[str]] = None,
                             title: Optional[str] = None,
                             time_slice: Optional[slice] = None,
                             start_date: Optional[str] = None,
                             data_manager: Optional[Any] = None) -> plt.Figure:
    """
    Plot multiple streamflow variables over time for a specific ensemble and HUC8.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset containing streamflow variables
    ensemble_id_idx : str or int
        The ensemble_id string or index to plot
    huc8_code : str or int, optional
        The HUC8 code to plot. If None, averages across all HUC8s.
    basin_name : str, optional
        If provided, subset to HUC8s in this basin using data_manager
    variables : list of str, optional
        List of variable names to plot. If None, plots streamflow_mean, streamflow_max, etc.
    title : str, optional
        Plot title. If None, creates a default title
    time_slice : slice, optional
        Time slice to plot (e.g., slice(0, 120) for first 120 time steps)
    start_date : str, optional
        Start date for time axis in format 'YYYY-MM' (e.g., '1980-01')
    data_manager : DataManager, optional
        DataManager instance needed if basin_name is provided
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object with the plot
    """
    # Handle ensemble ID - allow either string or index
    if isinstance(ensemble_id_idx, str):
        # Find the matching ensemble ID
        if ensemble_id_idx not in dataset["ensemble_id"].values:
            # Try to find a partial match
            matching_ids = [eid for eid in dataset["ensemble_id"].values if ensemble_id_idx in eid]
            if not matching_ids:
                raise ValueError(f"Ensemble ID '{ensemble_id_idx}' not found in dataset")
            ensemble_id_idx = matching_ids[0]
        
        # Get the index for this ensemble ID
        ensemble_idx = np.where(dataset["ensemble_id"].values == ensemble_id_idx)[0][0]
    else:
        # Use the provided index
        ensemble_idx = ensemble_id_idx
        ensemble_id_idx = dataset["ensemble_id"].values[ensemble_idx]
    
    # Extract ensemble metadata for the title
    ensemble_name = str(ensemble_id_idx)
    # Try to extract model and scenario from the ensemble ID string
    model_match = re.search(r'_([\w\-]+)_ssp', ensemble_name)
    model = model_match.group(1) if model_match else None
    
    scenario_match = re.search(r'_ssp(\d+)_', ensemble_name)
    scenario = f"ssp{scenario_match.group(1)}" if scenario_match else None
    
    # Identify numeric vars and subset
    numeric_vars = [var for var in dataset.data_vars 
                if np.issubdtype(dataset[var].dtype, np.number)]
    
    dataset = dataset[numeric_vars]
    
    # Handle basin subsetting if basin_name is provided
    if basin_name is not None:
        if data_manager is None:
            raise ValueError("data_manager must be provided when using basin_name")
        
        # Find HUC8s in the basin
        basin_huc8s = data_manager.find_huc8s_in_basin(basin_name)
        if basin_huc8s is None or len(basin_huc8s) == 0:
            raise ValueError(f"No HUC8s found for basin '{basin_name}'")
        
        # Get the HUC8 codes as integers
        basin_huc8_codes = basin_huc8s['HUC08'].astype(int).tolist()
        
        # Subset to only these HUC8s
        huc8_indices = [i for i, h in enumerate(dataset["huc8"].values) if h in basin_huc8_codes]
        if not huc8_indices:
            raise ValueError(f"None of the HUC8s in basin '{basin_name}' are found in the dataset")
        
        # Subset data to the basin HUC8s and take mean
        ds_subset = dataset.isel(ensemble_id=ensemble_idx, huc8=huc8_indices).mean(dim='huc8')
        region_name = f"{basin_name} Basin"
        
    # Handle individual HUC8 code if provided
    elif huc8_code is not None:
        # Convert to integer if it's a string
        if isinstance(huc8_code, str):
            huc8_code = int(huc8_code)
        
        # Find the index for this HUC8 code
        if huc8_code not in dataset["huc8"].values:
            raise ValueError(f"HUC8 code '{huc8_code}' not found in dataset")
        
        huc8_idx = np.where(dataset["huc8"].values == huc8_code)[0][0]
        
        # Subset data for this specific HUC8
        ds_subset = dataset.isel(ensemble_id=ensemble_idx, huc8=huc8_idx)
        region_name = f"HUC8: {huc8_code}"
    else:
        # Average across all HUC8s
        ds_subset = dataset.isel(ensemble_id=ensemble_idx).mean(dim='huc8')
        region_name = "All HUC8s (Average)"
    
    # Apply time slicing if provided
    if time_slice is not None:
        ds_subset = ds_subset.isel(time_mn=time_slice)
    
    # Create x-axis dates if start_date is provided
    if start_date is not None:
        try:
            start = datetime.strptime(start_date, '%Y-%m')
            # Create time axis with monthly increments
            time_values = [start + timedelta(days=30*i) for i in range(len(ds_subset["time_mn"]))]
            
            # Format x-axis based on length of time series
            if len(time_values) > 60:  # More than 5 years
                date_format = mdates.DateFormatter('%Y')
                locator = mdates.YearLocator(5)  # Every 5 years
            else:
                date_format = mdates.DateFormatter('%b %Y')
                locator = mdates.MonthLocator(interval=3)  # Every 3 months
                
            use_dates = True
        except ValueError:
            print(f"Warning: Could not parse start_date '{start_date}'. Using time indices instead.")
            time_values = ds_subset["time_mn"].values
            use_dates = False
            date_format = None
            locator = None
    else:
        time_values = ds_subset["time_mn"].values
        use_dates = False
        date_format = None
        locator = None
    
    # Set default variables to plot if not specified
    if variables is None:
        variables = ['streamflow_mean', 'streamflow_max', 'streamflow_min', 'streamflow_median', 'streamflow_std']
        # Filter to only include variables that exist in the dataset
        variables = [var for var in variables if var in dataset.data_vars]
    
    # Check if variables exist
    for var in variables:
        if var not in dataset.data_vars:
            raise ValueError(f"Variable '{var}' not found in dataset")
    
    # Create the figure with subplots
    fig, axes = plt.subplots(len(variables), 1, figsize=(12, 3*len(variables)), sharex=True)
    if len(variables) == 1:
        axes = [axes]  # Make axes always iterable
    
    # Plot each variable
    for i, var in enumerate(variables):
        ax = axes[i]
        
        # Plot the variable
        if use_dates:
            ax.plot(time_values, ds_subset[var].values)
        else:
            ds_subset[var].plot(ax=ax, label=var)
        
        # Add labels and styling
        var_name = var.replace('streamflow_', '')
        ax.set_title(f"Streamflow {var_name.capitalize()}")
        ax.set_ylabel(f"Flow ({getattr(ds_subset[var], 'units', 'CFS')})")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add mean line
        mean_value = ds_subset[var].mean().values
        ax.axhline(y=mean_value, color='r', linestyle='--', alpha=0.7)
        ax.text(0.02, 0.90, f'Mean: {mean_value:.2f}', transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Format the x-axis if we have date information
    if use_dates:
        for ax in axes:
            ax.xaxis.set_major_formatter(date_format)
            if locator:
                ax.xaxis.set_major_locator(locator)
        plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
        axes[-1].set_xlabel('Date')
    else:
        # Set common x-axis label
        axes[-1].set_xlabel('Time Index')
    
    # Add a title
    if title is None:
        if model and scenario:
            title_parts = [f"Streamflow for {model} {scenario}"]
        else:
            title_parts = [f"Streamflow for Ensemble {ensemble_name[:15]}..."]
        
        title_parts.append(region_name)
        
        title = " - ".join(title_parts)
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    return fig


def plot_seasonal_trends(dataset: xr.Dataset, ensemble_id_idx: Union[str, int],
                         huc8_code: Optional[Union[str, int]] = None,
                         basin_name: Optional[str] = None,
                         variable: str = 'streamflow_mean',
                         plot_type: str = 'monthly_avg',
                         time_groups: int = 4,
                         data_manager: Optional[Any] = None) -> plt.Figure:
    """
    Plot seasonal trends for a specific ensemble, HUC8, and variable.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset containing streamflow variables
    ensemble_id_idx : str or int
        The ensemble_id string or index to plot
    huc8_code : str or int, optional
        The HUC8 code to plot. If None, averages across all HUC8s.
    basin_name : str, optional
        If provided, subset to HUC8s in this basin using data_manager
    variable : str, default='streamflow_mean'
        The variable to plot
    plot_type : str, default='monthly_avg'
        Type of seasonal plot: 'monthly_avg', 'time_groups', or 'seasonal_pattern'
    time_groups : int, default=4
        Number of time groups to divide the data into for 'time_groups' plot_type
    data_manager : DataManager, optional
        DataManager instance needed if basin_name is provided
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object with the plot
    """
    # Handle ensemble ID - allow either string or index
    if isinstance(ensemble_id_idx, str):
        # Find the matching ensemble ID
        if ensemble_id_idx not in dataset["ensemble_id"].values:
            # Try to find a partial match
            matching_ids = [eid for eid in dataset["ensemble_id"].values if ensemble_id_idx in eid]
            if not matching_ids:
                raise ValueError(f"Ensemble ID '{ensemble_id_idx}' not found in dataset")
            ensemble_id_idx = matching_ids[0]
        
        # Get the index for this ensemble ID
        ensemble_idx = np.where(dataset["ensemble_id"].values == ensemble_id_idx)[0][0]
    else:
        # Use the provided index
        ensemble_idx = ensemble_id_idx
        ensemble_id_idx = dataset["ensemble_id"].values[ensemble_idx]
    
    # Check if variable exists
    if variable not in dataset.data_vars:
        raise ValueError(f"Variable '{variable}' not found in dataset")
    
    # Extract ensemble metadata for the title
    ensemble_name = str(ensemble_id_idx)
    # Try to extract model and scenario from the ensemble ID string
    model_match = re.search(r'_([\w\-]+)_ssp', ensemble_name)
    model = model_match.group(1) if model_match else None
    
    scenario_match = re.search(r'_ssp(\d+)_', ensemble_name)
    scenario = f"ssp{scenario_match.group(1)}" if scenario_match else None
    
    # Identify numeric vars and subset
    numeric_vars = [var for var in dataset.data_vars 
                if np.issubdtype(dataset[var].dtype, np.number)]
    
    dataset = dataset[numeric_vars]
    
    # Handle basin subsetting if basin_name is provided
    if basin_name is not None:
        if data_manager is None:
            raise ValueError("data_manager must be provided when using basin_name")
        
        # Find HUC8s in the basin
        basin_huc8s = data_manager.find_huc8s_in_basin(basin_name)
        if basin_huc8s is None or len(basin_huc8s) == 0:
            raise ValueError(f"No HUC8s found for basin '{basin_name}'")
        
        # Get the HUC8 codes as integers
        basin_huc8_codes = basin_huc8s['HUC08'].astype(int).tolist()
        
        # Subset to only these HUC8s
        huc8_indices = [i for i, h in enumerate(dataset["huc8"].values) if h in basin_huc8_codes]
        if not huc8_indices:
            raise ValueError(f"None of the HUC8s in basin '{basin_name}' are found in the dataset")
        
        # Subset data to the basin HUC8s and take mean
        ds_subset = dataset.isel(ensemble_id=ensemble_idx, huc8=huc8_indices).mean(dim='huc8')
        region_name = f"{basin_name} Basin"
        
    # Handle individual HUC8 code if provided
    elif huc8_code is not None:
        # Convert to integer if it's a string
        if isinstance(huc8_code, str):
            huc8_code = int(huc8_code)
        
        # Find the index for this HUC8 code
        if huc8_code not in dataset["huc8"].values:
            raise ValueError(f"HUC8 code '{huc8_code}' not found in dataset")
        
        huc8_idx = np.where(dataset["huc8"].values == huc8_code)[0][0]
        
        # Subset data for this specific HUC8
        ds_subset = dataset.isel(ensemble_id=ensemble_idx, huc8=huc8_idx)
        region_name = f"HUC8: {huc8_code}"
    else:
        # Average across all HUC8s
        ds_subset = dataset.isel(ensemble_id=ensemble_idx).mean(dim='huc8')
        region_name = "All HUC8s (Average)"
    
    # Get the variable data
    var_data = ds_subset[variable]
    
    # Create a figure
    if plot_type == 'monthly_avg':
        # Assume the time dimension is monthly data (since we have 480 = 40*12 time steps)
        # and we want to average over all years
        
        # Calculate monthly means (assuming 480 time steps = 40 years of monthly data)
        months_per_year = 12
        n_years = len(var_data["time_mn"]) // months_per_year
        
        # Reshape the data to (years, months)
        var_reshaped = var_data.values.reshape(n_years, months_per_year)
        
        # Calculate the mean for each month across all years
        monthly_means = np.mean(var_reshaped, axis=0)
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot monthly means
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.plot(range(1, 13), monthly_means, marker='o', linewidth=2)
        
        # Add labels and styling
        ax.set_xlabel('Month')
        ax.set_ylabel(f"{variable} ({getattr(var_data, 'units', 'CFS')})")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add a title
        if model and scenario:
            title_parts = [f"Monthly Average {variable.replace('streamflow_', 'Streamflow ')} for {model} {scenario}"]
        else:
            title_parts = [f"Monthly Average {variable.replace('streamflow_', 'Streamflow ')} for Ensemble {ensemble_name[:15]}..."]
        
        title_parts.append(region_name)
        
        ax.set_title(" - ".join(title_parts))
        
    elif plot_type == 'time_groups':
        # Divide the time series into groups and show the pattern for each group
        
        # Determine the number of time points in each group
        time_points = len(var_data["time_mn"])
        points_per_group = time_points // time_groups
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot each time group
        for i in range(time_groups):
            start_idx = i * points_per_group
            end_idx = (i + 1) * points_per_group if i < time_groups - 1 else time_points
            
            group_data = var_data.isel(time_mn=slice(start_idx, end_idx))
            
            # Calculate which time indices represent which months
            # Assuming 480 time steps = 40 years of monthly data
            months_per_year = 12
            month_indices = np.arange(points_per_group) % months_per_year
            
            # Create a new array of just the monthly data
            month_values = np.zeros(months_per_year)
            for m in range(months_per_year):
                month_mask = month_indices == m
                if np.any(month_mask):
                    month_values[m] = group_data.isel(time_mn=month_mask).mean().values
            
            # Plot this time group
            group_label = f"Period {i+1} ({start_idx//months_per_year+1}-{(end_idx-1)//months_per_year+1} yrs)"
            ax.plot(range(1, 13), month_values, marker='o', linewidth=2, label=group_label)
        
        # Add labels and styling
        ax.set_xlabel('Month')
        ax.set_ylabel(f"{variable} ({getattr(var_data, 'units', 'CFS')})")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title="Time Periods", loc='best')
        
        # Add a title
        if model and scenario:
            title_parts = [f"Seasonal Patterns by Period for {model} {scenario}"]
        else:
            title_parts = [f"Seasonal Patterns by Period for Ensemble {ensemble_name[:15]}..."]
        
        title_parts.append(region_name)
        
        ax.set_title(" - ".join(title_parts))
        
    elif plot_type == 'seasonal_pattern':
        # Create a seasonal decomposition plot
        
        # Assuming 480 time steps = 40 years of monthly data
        months_per_year = 12
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create a figure with 3 subplots (original, trend, seasonal)
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
        
        # Plot 1: Original time series
        ax_orig = axes[0]
        var_data.plot(ax=ax_orig)
        ax_orig.set_title("Original Time Series")
        ax_orig.set_xlabel("Time Index")
        ax_orig.set_ylabel(f"{variable} ({getattr(var_data, 'units', 'CFS')})")
        ax_orig.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Trend (moving average)
        ax_trend = axes[1]
        # Calculate 12-month moving average
        trend = np.convolve(var_data.values, np.ones(months_per_year)/months_per_year, mode='valid')
        ax_trend.plot(range(months_per_year-1, len(var_data["time_mn"])), trend)
        ax_trend.set_title("Trend (12-month Moving Average)")
        ax_trend.set_xlabel("Time Index")
        ax_trend.set_ylabel(f"{variable} ({getattr(var_data, 'units', 'CFS')})")
        ax_trend.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 3: Seasonal pattern
        ax_seasonal = axes[2]
        # Reshape the data to (years, months)
        n_years = len(var_data["time_mn"]) // months_per_year
        var_reshaped = var_data.values.reshape(n_years, months_per_year)
        
        # Calculate the mean for each month across all years
        seasonal_pattern = np.mean(var_reshaped, axis=0)
        
        ax_seasonal.plot(range(1, 13), seasonal_pattern, marker='o', linewidth=2)
        ax_seasonal.set_title("Seasonal Pattern (Average by Month)")
        ax_seasonal.set_xlabel("Month")
        ax_seasonal.set_ylabel(f"{variable} ({getattr(var_data, 'units', 'CFS')})")
        ax_seasonal.set_xticks(range(1, 13))
        ax_seasonal.set_xticklabels(month_names)
        ax_seasonal.grid(True, linestyle='--', alpha=0.7)
        
        # Overall title
        if model and scenario:
            title = f"Seasonal Decomposition for {model} {scenario}"
        else:
            title = f"Seasonal Decomposition for Ensemble {ensemble_name[:15]}..."
        
        title += f" - {region_name}"
        
        fig.suptitle(title, fontsize=16)
        
    else:
        raise ValueError(f"Unsupported plot_type: {plot_type}. Use 'monthly_avg', 'time_groups', or 'seasonal_pattern'")
    
    plt.tight_layout()
    if plot_type == 'seasonal_pattern':
        plt.subplots_adjust(top=0.93)
    
    return fig

import h5py
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
from os.path import join
from toolkit import repo_data_path
import yaml
from typing import Dict, Any, Optional


def dict_to_hdf5(filepath, data_dictionary):
    with h5py.File(filepath, "w") as f:
        for key, value in data_dictionary.items():
            f.create_dataset(key, data=value)

def hdf5_to_dict(filepath): 
    data_dict = dict()
    with h5py.File(filepath) as f:
        for key in f.keys():
            if f[key].dtype == "O":
                data_dict[key] = f[key].asstr()[:]
            elif f[key].dtype == "f8" and key=="shortage_data":
                data_dict[key] = np.array(f[key], dtype=np.float32)
            else:    
                data_dict[key] = f[key][:]
    return data_dict

def load_right_latlongs(latlong_gdf_path=None):
    if latlong_gdf_path is None:
        latlong_gdf_path = join(repo_data_path, "geospatial", "right_latlongs.geojson")
    latlongs = gpd.read_file(latlong_gdf_path)
    latlongs.set_index("water_right_identifier", inplace=True)
    return latlongs

def load_crb_shape(crb_path=None):
    if crb_path is None:
        crb_path = join(repo_data_path, "geospatial", "CRB")
    crb = gpd.read_file(crb_path)
    return crb

def load_config(file):
    with open(file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config



def convert_to_netcdf_format(data_dictionary: Dict[str, Any], 
                           additional_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convert the dictionary output from generate_synthetic_streamflow to NetCDF-compatible format.
    
    Parameters
    ----------
    data_dictionary : Dict[str, Any]
        Dictionary output from generate_synthetic_streamflow method
    additional_metadata : Optional[Dict[str, Any]], default=None
        Additional metadata to include in the output
        
    Returns
    -------
    Dict[str, Any]
        NetCDF-compatible dictionary format
    """
    # Extract data from input dictionary
    streamflow_data = data_dictionary['streamflow']
    annual_states_data = data_dictionary['annual_states']
    realization_meta = data_dictionary['realization_meta']
    realization_meta_labels = data_dictionary['realization_meta_labels']
    streamflow_index = data_dictionary['streamflow_index']
    streamflow_columns = data_dictionary['streamflow_columns']
    annual_states_index = data_dictionary['annual_states_index']
    
    # Get dimensions
    n_realizations, n_months, n_sites = streamflow_data.shape
    n_years = annual_states_data.shape[1]
    
    # Convert to lists
    time_index = list(streamflow_index)
    site_names = list(streamflow_columns)
    year_index = list(annual_states_index)
    
    start_year = int(time_index[0].split('-')[0])
    end_year = int(time_index[-1].split('-')[0])
    
    # Get basin name and filter name from additional metadata
    basin_name = additional_metadata['basin_name']
    subset_name = additional_metadata['subset_name']
    
    # Create NetCDF-compatible format
    netcdf_dict = {
        # Data variables
        'synthetic_streamflow': {
            'data': streamflow_data,
            'dims': ['realization', 'time_step', 'gage_id'],
            'attrs': {
                'units': 'acre-feet',
                'description': 'Monthly synthetic streamflow generated from Bayesian HMM',
            }
        },
        'annual_wet_dry_state': {
            'data': annual_states_data,
            'dims': ['realization', 'year'],
            'attrs': {
                'description': 'HMM hidden states generated for each year and realization. 0 is dry, 1 is wet.',
                'valid_range': [0, 1]
            }
        },
        
        # HMM parameters as data variables
        'hmm_parameters': {
            'data': realization_meta,
            'dims': ['realization', 'hmm_parameter_name'],
            'attrs': {
                'long_name': 'HMM model parameters',
                'description': 'Hidden Markov Model parameters for each realization',
            }
        },
        
        # Coordinate variables
        'realization': {
            'data': np.arange(n_realizations),
            'attrs': {
                'long_name': 'Realization index',
                'description': 'Index for each synthetic realization',
                'units': 'integer ordering'
            }
        },
        'time_step': {
            'data': pd.to_datetime(time_index),
            'attrs': {
                'long_name': 'Time step',
                'description': 'Monthly time steps (YYYY-MM-DD)'
            }
        },
        'gage_id': {
            'data': np.array(site_names, dtype='U'),
            'attrs': {
                'long_name': 'Gage site names',
                'description': 'Names of streamflow gages used by WRAP'
            }
        },
        'year': {
            'data': np.array(year_index, dtype='U'),
            'attrs': {
                'long_name': 'Year',
                'description': 'Year labels for annual_wet_dry_state',
            }
        },
        'hmm_parameter_name': {
            'data': np.array(realization_meta_labels),
            'attrs': {
                'long_name': 'Parameter labels',
                'description': 'Labels of HMM parameters',
            }
        },
        
        # Global attributes
        'global_attrs': {
            'title': f'Synthetic Streamflow Realizations for {basin_name} generated using {subset_name} subset',
            'source': 'Bayesian Hidden Markov Model',
            'creation_date': pd.Timestamp.now().isoformat(),
            'n_realizations': n_realizations,
            'n_months': n_months,
            'n_gages': n_sites,
            'n_years': n_years,
            'start_year': start_year,
            'end_year': end_year,
            'n_hmm_parameters': len(realization_meta_labels),
            'temporal_resolution': 'monthly',
            'spatial_resolution': 'streamflow gage sites',
            'generation_method': 'HMM with historical disaggregation',
        }
    }
    
    # Add additional metadata if provided
    if additional_metadata:
        netcdf_dict['global_attrs'].update(additional_metadata)
    
    return netcdf_dict


def save_netcdf_format(data_dictionary: Dict[str, Any], 
                      output_path: str,
                      additional_metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Convert dictionary to NetCDF format and save to file.
    
    Parameters
    ----------
    data_dictionary : Dict[str, Any]
        Dictionary output from generate_synthetic_streamflow method
    output_path : str
        Path to save the NetCDF file
    additional_metadata : Optional[Dict[str, Any]], default=None
        Additional metadata to include
    """
    import xarray as xr
    
    # Convert to NetCDF format
    netcdf_dict = convert_to_netcdf_format(
        data_dictionary, additional_metadata
    )
    
    # Create xarray Dataset
    data_vars = {}
    coords = {}
    
    # Add data variables
    for var_name, var_info in netcdf_dict.items():
        if var_name == 'global_attrs':
            continue

        if 'dims' in var_info:
            # Data variable
            data_vars[var_name] = (
                var_info['dims'],
                var_info['data'],
                var_info.get('attrs', {})
            )
        else:
            # Coordinate variable
            coords[var_name] = (
                var_name,
                var_info['data'],
                var_info.get('attrs', {})
            )
    
    # Create Dataset
    ds = xr.Dataset(data_vars, coords=coords, attrs=netcdf_dict['global_attrs'])
    
    # Save to NetCDF
    ds.to_netcdf(output_path)
    
    return ds

def load_netcdf_format(filepath: str) -> dict:
    """
    Load a NetCDF file created by save_netcdf_format and reconstruct
    the original dictionary structure (streamflow, annual_states, etc.).

    Parameters
    ----------
    filepath : str
        Path to the NetCDF file

    Returns
    -------
    dict
        Dictionary in the same format used when creating the NetCDF
    """
    with xr.open_dataset(filepath) as ds:
        # Extract data variables
        streamflow_out = ds['synthetic_streamflow'].values
        annual_states = ds['annual_wet_dry_state'].values
        hmm_params = ds['hmm_parameters'].values

        # Extract coordinates (decode to str for consistency with input dict)
        time_index = ds['time_step'].values.astype('datetime64[M]').astype(str).tolist()
        site_names = ds['gage_id'].values.astype(str).tolist()
        year_index = ds['year'].values.astype(str).tolist()
        hmm_param_labels = ds['hmm_parameter_name'].values.astype(str).tolist()
        if isinstance(hmm_param_labels, np.ndarray):
            hmm_param_labels = hmm_param_labels.tolist()

        # Rebuild dictionary in the same structure as input
        data_dictionary = {
            'streamflow': streamflow_out,
            'annual_states': annual_states,  # already (realization, year)
            'realization_meta': hmm_params,
            'realization_meta_labels': np.array(hmm_param_labels, dtype='U'),
            'streamflow_index': np.array(time_index, dtype='U'),
            'streamflow_columns': np.array(site_names, dtype='U'),
            'annual_states_index': np.array(year_index, dtype='U')
        }

        return data_dictionary
    
    
# Data Documentation

This dataset contains synthetically generated streamflow realizations paired with corresponding water management outputs from the Water Rights Analysis Package (WRAP) for several river basins in Texas.

The dataset is generated using a Bayesian Hidden Markov Model (BHMM) trained on the [DOE 9505 streamflow projection ensemble](https://hydrosource.ornl.gov/data/datasets/9505v3_1/). A set of 1000 streamflow realizations and corresponding outputs from WRAP are generated for three river basins in texas using seven different subsets of the 9505 ensemble.

Basins:
- Colorado River Basin
- Trinity River Basin
- Trinity River Basin

9505 Subsets:
- All Models (All 48 ensemble members)
- Bias Correction - Daymet (The 24 ensemble members which use Daymet bias correction)
- Bias Correction - Livneh (The 24 ensemble members which use Livneh bias correction)
- Downscaling - DBCCA (The 24 ensemble members which use DBCCA downscaling )
- Downscaling - RegCM (The 24 ensemble members which use RegCM downscaling)
- Hydro Model - PRMS (The 24 ensemble members which use the PRMS hydrological model)
- Hydro Model - VIC5 (The 24 ensemble members which use the VIC5 hydrological model)
- Bias Correction - Dayment (The 24 ensemble members which use Daymet bias correction)

For each pair of basin and subset (21 total), the following is performed:
1. A an annual BHMM is fit to the outlet streamflow for the basin using the ensembles of the subset.
2. A set of 1000 streamflow realizations is generated using the fit BHMM.
3. The streamflow realizations are simulated in WRAP to obtain water management outputs.
4. The streamflow realizations and water management outputs are bundled into a single NetCDF.

The core files of this archive are the 21 NetCDF files generated from this process. They are organized into the following file structure TODO. In addition to the core files, there is a `data` folder which contains the data necessary for reproducing the dataset. The details of these two parts of the archive are provided below


## NetCDF File Structure

The NetCDF files containing synthetic streamflow and water management outputs:

### File: `{9505_subset}_{basin}_synthetic_dataset.nc`

#### Dimensions
- **`realization`**: Integer identifier for individual realizations
- **`time_step`**: Monthly time steps (e.g., 1940-01-01 to 2016-12-01)
- **`gage_id`**: Streamflow gage sites (e.g., INA10000, INA20000, etc.)
- **`year`**: Annual time steps for hidden states
- **`hmm_parameter_name`**: HMM parameter labels
- **`right_id`**: Water right identifiers for diversion data
- **`reservoir_id`**: Reservoir identifiers for reservoir data

#### Data Variables

##### 1. `synthetic_streamflow`
- **Description**: Monthly synthetic streamflow generated from Bayesian HMM.
- **Dimensions**: `[realization, time_step site]`
- **Units**: acre-feet

##### 2. `annual_wet_dry_state`
- **Description**: HMM hidden states for each year and realization. 0 is dry, 1 is wet. For example if the hidden state at realization 2 and year 2000 is 1, that means the streamflow for that year in that specific realization was emitted from the wet state distribution.
- **Dimensions**: `[realization, year]`
- **Values**: 0 (dry state) or 1 (wet state)

##### 3. `hmm_parameters`
- **Description**: Hidden Markov Model parameters used to generate each realization
- **Dimensions**: `[realization, hmm_parameter_name]`

##### 4. `diversion_or_energy_shortage`
- **Description**: Water shortage from WRAP model simulation.
- **Dimensions**: `[realization, time_step right_id]`
- **Units**: acre-feet

##### 6. `diversion_or_energy_target`
- **Description**: Target water allocation from WRAP model simulation
- **Dimensions**: `[realization, time_step right_id]`
- **Units**: acre-feet

##### 7. `shortage_ratio`
- **Description**: Water shortage ratio from WRAP model simulation (1 - (target - shortage) / target). 0 means no shortage, 1 means full shortage.
- **Dimensions**: `[realization, time_step right_id]`
- **Units**: ratio [0-1]

##### 8. `reservoir_water_surface_elevation`
- **Description**: Water surface elevation of reservoirs from WRAP model simulation.
- **Dimensions**: `[realization, time_step reservoir_id]`
- **Units**: feet (unverified)

##### 9. `reservoir_storage_capacity`
- **Description**: Storage capacity of reservoirs from WRAP model simulation.
- **Dimensions**: `[realization, time_step reservoir_id]`
- **Units**: acre-feet (unverified)

##### 10. `reservoir_inflows_to_reservoir_from_stream_flow_depletions`
- **Description**: Inflows to reservoirs from stream flow depletions from WRAP model simulation.
- **Dimensions**: `[realization, time_step reservoir_id]`
- **Units**: acre-feet (unverified)

##### 11. `reservoir_inflows_to_reservoir_from_releases_from_other_reservoirs`
- **Description**: Inflows to reservoirs from releases of other reservoirs from WRAP model simulation.
- **Dimensions**: `[realization, time_step reservoir_id]`
- **Units**: acre-feet (unverified)

##### 12. `reservoir_net_evaporation_precipitation_volume`
- **Description**: Net evaporation and precipitation volume for reservoirs from WRAP model simulation
- **Dimensions**: `[realization, time_step reservoir_id]`
- **Units**: acre-feet (unverified)

##### 13. `reservoir_energy_generated`
- **Description**: Energy generated from hydroelectric power from WRAP model simulation.
- **Dimensions**: `[realization, time_step reservoir_id]`
- **Units**: MWh (unverified)

##### 14. `reservoir_releases_accessible_to_hydroelectric_power_turbines`
- **Description**: Reservoir releases accessible to hydroelectric power turbines from WRAP model simulation.
- **Dimensions**: `[realization, time_step reservoir_id]`
- **Units**: acre-feet (unverified)

##### 15. `reservoir_releases_not_accessible_to_hydroelectric_power_turbines`
- **Description**: Reservoir releases not accessible to hydroelectric power turbines from WRAP model simulation.
- **Dimensions**: `[realization, time_step reservoir_id]`
- **Units**: acre-feet (unverified)

#### Coordinate Variables

##### 1. `realization`
- **Description**: Index for each synthetic realization.
- **Values**: [0, 1, 2, ..., n_realizations-1]

##### 2. `time_step`
- **Description**: Monthly time steps (YYYY-MM-DD).
- **Values**: Datetime strings (e.g., 1940-01-01, 1940-02-01, ...)

##### 3. `gage_id`
- **Description**: Index of streamflow gage IDs used by WRAP.
- **Values**: Gage identifiers (e.g., INA10000, INA20000, INA30000, ...)

##### 4. `year`
- **Description**: Year labels for annual states
- **Values**: Year strings (e.g., "1940", "1941", "1942", ...)

##### 5. `parameter`
- **Description**: Labels of HMM parameters
- **Values**: Parameter names (e.g., transition probabilities, emission parameters)

##### 6. `right_id`
- **Description**: Index of water right identifiers used by WRAP.
- **Values**: Water right IDs from WRAP model

##### 7. `reservoir_id`
- **Description**: Index of reservoir identifiers used by WRAP.
- **Values**: Reservoir IDs from WRAP model


## `data` folder

The `data` folder contains all the necessary files to reproduce this dataset:

### Directory Structure
```
data/
├── configs/
│   ├── basins.json - Metadata for the basins used in the experiment.
│   ├── ensemble_filters.json - Filters for 9505 ensemble subsets used in the experiment.
│   ├── random_seeds.json - Random seeds used for reproducibility across multiple steps of the experiment.
│   ├── reaches_of_interest.csv - Reaches extracted from the 9505 data for downstream use in analysis.
│   └── wrap_variable_metadata.json - Descriptive metadata for the water management outputs of WRAP
├── geospatial/
│   ├── 9505_shapefiles/ - Shapefiles related to the DOE 9505 dataset.
│   └── wrap_gages/ - Shapefiles containing primary control point (gage) locations used by WRAP for each basin.
└── WRAP/
    ├── basin_wams/ - Water Availability Models for use with WRAP.
    └── SIM.exe - WRAP executable file.
```

### Configuration Files

**`basins.json`** contains metadata for the basins. Currently, four attributes are provided:
- `gage_name`: The name of the WRAP control point at the outflow gage.
- `reach_id`: The name of the reach in the 9505 data which has been associated to the outflow gage.
- `flo_file`: The path to the .FLO file associated to the basin (used in WRAP simulations).

**`ensemble_filters.json`** and **`ensemble_filters_basic.json`** define the different subsets of the 9505 ensemble used for training the BHMM models.

**`wrap_variable_metadata.json`** contains metadata for all WRAP output variables, including units, descriptions, and names for diversion and reservoir variables.

**`reaches_of_interest.csv`** lists the specific river reaches (COMIDs) extracted from the 9505 dataset for each basin.

### Geospatial Data

**`9505_shapefiles/`** contains the National Hydrography Dataset (NHD) flowline shapefiles for HUC2 regions 11, 12, and 13, used for associating streamflow reaches with control points. Other data provided by the 9505 datset is also included, but not needed for the reproducibility of this dataset.

**`wrap_gages/`** contains shapefiles with the primary control point locations for each basin, used to identify the outlet gages for WRAP simulations.

### WRAP Model Files

**`basin_wams/`** contains the Water Availability Model (WAM) files for each basin, including:
- Basin configuration files
- Water right data
- Reservoir data
- Diversion data

**`SIM.exe`** is the WRAP simulation executable (Windows binary, run using Wine on Linux).

**`wrap_execution_directories/`** contains the execution directories used for running WRAP simulations with different streamflow inputs.

## Usage Examples

### Load NetCDF file in Python:
```python
import xarray as xr

# Load the dataset
ds = xr.open_dataset('colorado_synthetic_dataset.nc')

# Access streamflow data
streamflow = ds['synthetic_streamflow'] # [realization, time_step site]

# Access shortage data
shortage = ds['shortage_ratio'] # [realization, time_step right_id]

# Access annual states
annual_states = ds['annual_wet_dry_state'] # [realization, year]

# Access HMM parameters
hmm_params = ds['hmm_parameters'] # [realization, parameter]

# Access diversion data
diversion_shortage = ds['diversion_or_energy_shortage'] # [realization, time_step right_id]
diversion_target = ds['diversion_or_energy_target'] # [realization, time_step right_id]
shortage_ratio = ds['diversion_shortage_ratio'] # [realization, time_step right_id]

# Access reservoir data
reservoir_elevation = ds['reservoir_water_surface_elevation'] # [realization, time_step reservoir_id]
reservoir_capacity = ds['reservoir_storage_capacity'] # [realization, time_step reservoir_id]
energy_generated = ds['reservoir_energy_generated'] # [realization, time_step reservoir_id]
```

## Scripts
`explore_netcdf.py` walks through the opening of the NetCDF file and exploration of its contents.

# Ensemble Data Documentation

This dataset contains an ensemble of synthetically generated streamflow ensembles paired with corresponding water management outputs from the Water Rights Analysis Package (WRAP) for several river basins in Texas.

The core dataset is provided in two formats. The first is a fully annotated NetCDF file. The second is a set of csv files which provide the primary data outputs in indvidual csv files for each ensemble member. 

Additional files include a JSON file containing basic metadata about the basins and other data required to reproduce this dataset.

## NetCDF File Structure

The main data file is a NetCDF file containing synthetic streamflow and shortage data:

### File: `{basin_name}_9505_hmm_ensemble.nc`

#### Dimensions
- **`ensemble`**: Integer identifier for individual ensemble members
- **`time`**: Monthly time steps (e.g., 1940-01-01 to 2016-12-01)
- **`site`**: Streamflow gage sites (e.g., INA10000, INA20000, etc.)
- **`year`**: Annual time steps for hidden states
- **`parameter`**: HMM parameter labels
- **`right_id`**: Water right identifiers for diversion data
- **`reservoir_id`**: Reservoir identifiers for reservoir data

#### Data Variables

##### 1. `streamflow`
- **Description**: Synthetic monthly streamflow data
- **Dimensions**: `[ensemble, time, site]`
- **Units**: acre-feet
- **Attributes**:
  - `long_name`: "Synthetic streamflow"
  - `description`: "Monthly synthetic streamflow generated from Bayesian HMM"
  - `standard_name`: "streamflow"

##### 2. `shortage`
- **Description**: Water shortage ratios from WRAP simulations
- **Dimensions**: `[ensemble, time, right_id]`
- **Units**: ratio [0-1]
- **Attributes**:
  - `long_name`: "Water shortage ratio"
  - `description`: "Water shortage ratio from WRAP model simulation (1 - (target - shortage) / target). 0 means no shortage, 1 means full shortage."
  - `standard_name`: "shortage_ratio"

##### 3. `annual_states`
- **Description**: HMM hidden states for each year
- **Dimensions**: `[ensemble, year]`
- **Values**: 0 (dry state) or 1 (wet state)
- **Attributes**:
  - `long_name`: "Annual hidden states"
  - `description`: "HMM hidden states for each year and ensemble member. 0 is dry, 1 is wet. For example if the hidden state at ensemble 2 and year 2000 is 1, that means the streamflow for that year in that specific ensemble was emitted from the wet state distribution."
  - `standard_name`: "hidden_state"
  - `valid_range`: [0, 1]

##### 4. `hmm_parameters`
- **Description**: HMM model parameters for each ensemble member
- **Dimensions**: `[ensemble, parameter]`
- **Attributes**:
  - `long_name`: "HMM model parameters"
  - `description`: "Hidden Markov Model parameters for each ensemble member"

##### 5. `diversion_diversion_or_energy_shortage`
- **Description**: Water shortage for diversions or energy generation from WRAP model simulation
- **Dimensions**: `[ensemble, time, right_id]`
- **Units**: acre-feet
- **Attributes**:
  - `long_name`: "Diversion Or Energy Shortage"
  - `description`: "Water shortage for diversions or energy generation from WRAP model simulation"
  - `standard_name`: "diversion_shortage"

##### 6. `diversion_diversion_or_energy_target`
- **Description**: Target water allocation for diversions or energy generation from WRAP model simulation
- **Dimensions**: `[ensemble, time, right_id]`
- **Units**: acre-feet
- **Attributes**:
  - `long_name`: "Diversion Or Energy Target"
  - `description`: "Target water allocation for diversions or energy generation from WRAP model simulation"
  - `standard_name`: "diversion_target"

##### 7. `diversion_shortage_ratio`
- **Description**: Water shortage ratio from WRAP model simulation
- **Dimensions**: `[ensemble, time, right_id]`
- **Units**: ratio [0-1]
- **Attributes**:
  - `long_name`: "Water Shortage Ratio"
  - `description`: "Water shortage ratio from WRAP model simulation (1 - (target - shortage) / target). 0 means no shortage, 1 means full shortage"
  - `standard_name`: "shortage_ratio"

##### 8. `reservoir_reservoir_water_surface_elevation`
- **Description**: Water surface elevation of reservoirs from WRAP model simulation
- **Dimensions**: `[ensemble, time, reservoir_id]`
- **Units**: feet
- **Attributes**:
  - `long_name`: "Reservoir Water Surface Elevation"
  - `description`: "Water surface elevation of reservoirs from WRAP model simulation"
  - `standard_name`: "reservoir_elevation"

##### 9. `reservoir_reservoir_storage_capacity`
- **Description**: Storage capacity of reservoirs from WRAP model simulation
- **Dimensions**: `[ensemble, time, reservoir_id]`
- **Units**: acre-feet
- **Attributes**:
  - `long_name`: "Reservoir Storage Capacity"
  - `description`: "Storage capacity of reservoirs from WRAP model simulation"
  - `standard_name`: "reservoir_capacity"

##### 10. `reservoir_inflows_to_reservoir_from_stream_flow_depletions`
- **Description**: Inflows to reservoirs from stream flow depletions from WRAP model simulation
- **Dimensions**: `[ensemble, time, reservoir_id]`
- **Units**: acre-feet
- **Attributes**:
  - `long_name`: "Reservoir Inflows From Stream Flow Depletions"
  - `description`: "Inflows to reservoirs from stream flow depletions from WRAP model simulation"
  - `standard_name`: "reservoir_inflow_depletions"

##### 11. `reservoir_inflows_to_reservoir_from_releases_from_other_reservoirs`
- **Description**: Inflows to reservoirs from releases of other reservoirs from WRAP model simulation
- **Dimensions**: `[ensemble, time, reservoir_id]`
- **Units**: acre-feet
- **Attributes**:
  - `long_name`: "Reservoir Inflows From Other Reservoir Releases"
  - `description`: "Inflows to reservoirs from releases of other reservoirs from WRAP model simulation"
  - `standard_name`: "reservoir_inflow_releases"

##### 12. `reservoir_reservoir_net_evaporation_precipitation_volume`
- **Description**: Net evaporation and precipitation volume for reservoirs from WRAP model simulation
- **Dimensions**: `[ensemble, time, reservoir_id]`
- **Units**: acre-feet
- **Attributes**:
  - `long_name`: "Reservoir Net Evaporation Precipitation Volume"
  - `description`: "Net evaporation and precipitation volume for reservoirs from WRAP model simulation"
  - `standard_name`: "reservoir_net_evap_precip"

##### 13. `reservoir_energy_generated`
- **Description**: Energy generated from hydroelectric power from WRAP model simulation
- **Dimensions**: `[ensemble, time, reservoir_id]`
- **Units**: MWh
- **Attributes**:
  - `long_name`: "Energy Generated"
  - `description`: "Energy generated from hydroelectric power from WRAP model simulation"
  - `standard_name`: "energy_generated"

##### 14. `reservoir_reservoir_releases_accessible_to_hydroelectric_power_turbines`
- **Description**: Reservoir releases accessible to hydroelectric power turbines from WRAP model simulation
- **Dimensions**: `[ensemble, time, reservoir_id]`
- **Units**: acre-feet
- **Attributes**:
  - `long_name`: "Reservoir Releases Accessible To Hydroelectric Power Turbines"
  - `description`: "Reservoir releases accessible to hydroelectric power turbines from WRAP model simulation"
  - `standard_name`: "reservoir_releases_turbines"

##### 15. `reservoir_reservoir_releases_not_accessible_to_hydroelectric_power_turbines`
- **Description**: Reservoir releases not accessible to hydroelectric power turbines from WRAP model simulation
- **Dimensions**: `[ensemble, time, reservoir_id]`
- **Units**: acre-feet
- **Attributes**:
  - `long_name`: "Reservoir Releases Not Accessible To Hydroelectric Power Turbines"
  - `description`: "Reservoir releases not accessible to hydroelectric power turbines from WRAP model simulation"
  - `standard_name`: "reservoir_releases_non_turbines"

#### Coordinate Variables

##### 1. `ensemble`
- **Description**: Ensemble member index
- **Values**: [0, 1, 2, ..., n_ensembles-1]

##### 2. `time`
- **Description**: Monthly time steps
- **Values**: Datetime strings (e.g., 1940-01-01, 1940-02-01, ...)
- **Format**: YYYY-MM-DD

##### 3. `site`
- **Description**: Streamflow gage site names
- **Values**: Site identifiers (e.g., INA10000, INA20000, INA30000, ...)

##### 4. `year`
- **Description**: Year labels for annual states
- **Values**: Year strings (e.g., "1940", "1941", "1942", ...)

##### 5. `parameter`
- **Description**: HMM parameter labels
- **Values**: Parameter names (e.g., transition probabilities, emission parameters)

##### 6. `right_id`
- **Description**: Water right identifiers
- **Values**: Water right IDs from WRAP model

##### 7. `reservoir_id`
- **Description**: Reservoir identifiers
- **Values**: Reservoir IDs from WRAP model

## CSV File Structure

The extraction scripts produce organized CSV files for each ensemble member:

### Directory Structure
```
outputs/ensemble_csvs/{basin_name}/
├── streamflow/
│   ├── streamflow_ensemble_00.csv
│   ├── streamflow_ensemble_01.csv
│   └── ...
├── shortage/
│   ├── shortage_ensemble_00.csv
│   ├── shortage_ensemble_01.csv
│   └── ...
├── annual_states/
│   ├── annual_states_ensemble_00.csv
│   ├── annual_states_ensemble_01.csv
│   └── ...
├── hmm_parameters/
│   ├── hmm_parameters_ensemble_00.csv
│   ├── hmm_parameters_ensemble_01.csv
│   └── ...
├── diversions/
│   ├── diversion_diversion_or_energy_shortage_ensemble_00.csv
│   ├── diversion_diversion_or_energy_shortage_ensemble_01.csv
│   ├── diversion_diversion_or_energy_target_ensemble_00.csv
│   ├── diversion_diversion_or_energy_target_ensemble_01.csv
│   ├── diversion_shortage_ratio_ensemble_00.csv
│   ├── diversion_shortage_ratio_ensemble_01.csv
│   └── ...
└── reservoirs/
    ├── reservoir_reservoir_water_surface_elevation_ensemble_00.csv
    ├── reservoir_reservoir_water_surface_elevation_ensemble_01.csv
    ├── reservoir_reservoir_storage_capacity_ensemble_00.csv
    ├── reservoir_reservoir_storage_capacity_ensemble_01.csv
    ├── reservoir_energy_generated_ensemble_00.csv
    ├── reservoir_energy_generated_ensemble_01.csv
    └── ...
```

### 1. Streamflow CSV Files
**File**: `streamflow_ensemble_{XX}.csv`

**Columns**:
- `time`: Monthly timestamps (YYYY-MM-DD format)
- `{site_name}`: One column for each streamflow gage site

**Example**:
```csv
time,INA10000,INA20000,INA30000,INB10000,INB20000,...
1940-01-01,0.0,0.0,0.0,0.0,0.0,...
1940-02-01,0.0,0.0,0.0,0.0,0.0,...
1940-03-01,0.0,0.0,0.0,0.0,0.0,...
```

### 2. Shortage CSV Files
**File**: `shortage_ensemble_{XX}.csv`

**Columns**:
- `time`: Monthly timestamps (YYYY-MM-DD format)
- `{right_id}`: One column for each water right identifier

**Example**:
```csv
time,WR001,WR002,WR003,WR004,WR005,...
1940-01-01,0.0,0.1,0.0,0.2,0.0,...
1940-02-01,0.0,0.0,0.0,0.1,0.0,...
1940-03-01,0.0,0.0,0.0,0.0,0.0,...
```

### 3. Annual States CSV Files
**File**: `annual_states_ensemble_{XX}.csv`

**Columns**:
- `year`: Year labels (string format)
- `annual_state`: HMM hidden state (0 = dry, 1 = wet)

**Example**:
```csv
year,annual_state
1940,0
1941,1
1942,0
1943,1
1944,0
```

### 4. HMM Parameters CSV Files
**File**: `hmm_parameters_ensemble_{XX}.csv`

**Columns**:
- `parameter`: Parameter name/label
- `value`: Parameter value

**Example**:
```csv
parameter,value
transition_prob_00,0.85
transition_prob_01,0.15
transition_prob_10,0.20
transition_prob_11,0.80
emission_mean_0,1000.5
emission_std_0,250.3
emission_mean_1,2000.8
emission_std_1,400.2
```

### 5. Diversion CSV Files
**File**: `diversion_{variable_name}_ensemble_{XX}.csv`

**Columns**:
- `time`: Monthly timestamps (YYYY-MM-DD format)
- `{right_id}`: One column for each water right identifier

**Example** (diversion_shortage_ratio_ensemble_00.csv):
```csv
time,WR001,WR002,WR003,WR004,WR005,...
1940-01-01,0.0,0.1,0.0,0.2,0.0,...
1940-02-01,0.0,0.0,0.0,0.1,0.0,...
1940-03-01,0.0,0.0,0.0,0.0,0.0,...
```

### 6. Reservoir CSV Files
**File**: `reservoir_{variable_name}_ensemble_{XX}.csv`

**Columns**:
- `time`: Monthly timestamps (YYYY-MM-DD format)
- `{reservoir_id}`: One column for each reservoir identifier

**Example** (reservoir_reservoir_water_surface_elevation_ensemble_00.csv):
```csv
time,RES001,RES002,RES003,RES004,RES005,...
1940-01-01,1200.5,1150.2,1100.8,1250.1,1180.3,...
1940-02-01,1198.2,1148.9,1098.5,1248.8,1178.1,...
1940-03-01,1195.8,1147.3,1096.1,1247.2,1175.8,...
```

## Basin Metadata JSON
`basins.json` contains metadata for the basins. Currently, three attributes are provided:
- `gage_name`: The name of the WRAP control point at the outflow gage.
- `reach_id`: The name of the reach in the 9505 data which has been associated to the outflow gage.
- `flo_file`: The path to the .FLO file associated to the basin (this can be ignored outside of the main workflows).

## Usage Examples

### Load NetCDF file in Python:
```python
import xarray as xr

# Load the dataset
ds = xr.open_dataset('colorado_synthetic_streamflow.nc')

# Access streamflow data
streamflow = ds['streamflow']  # [ensemble, time, site]

# Access shortage data
shortage = ds['shortage']      # [ensemble, time, right_id]

# Access annual states
annual_states = ds['annual_states']  # [ensemble, year]

# Access HMM parameters
hmm_params = ds['hmm_parameters']    # [ensemble, parameter]

# Access diversion data
diversion_shortage = ds['diversion_diversion_or_energy_shortage']  # [ensemble, time, right_id]
diversion_target = ds['diversion_diversion_or_energy_target']      # [ensemble, time, right_id]
shortage_ratio = ds['diversion_shortage_ratio']                    # [ensemble, time, right_id]

# Access reservoir data
reservoir_elevation = ds['reservoir_reservoir_water_surface_elevation']  # [ensemble, time, reservoir_id]
reservoir_capacity = ds['reservoir_reservoir_storage_capacity']          # [ensemble, time, reservoir_id]
energy_generated = ds['reservoir_energy_generated']                      # [ensemble, time, reservoir_id]
```

### Load CSV files in Python:
```python
import pandas as pd

# Load streamflow data for ensemble 0
streamflow_df = pd.read_csv('streamflow_ensemble_00.csv', parse_dates=['time'])

# Load shortage data for ensemble 0
shortage_df = pd.read_csv('shortage_ensemble_00.csv', parse_dates=['time'])

# Load annual states for ensemble 0
annual_states_df = pd.read_csv('annual_states_ensemble_00.csv')

# Load HMM parameters for ensemble 0
hmm_params_df = pd.read_csv('hmm_parameters_ensemble_00.csv')

# Load diversion data for ensemble 0
diversion_shortage_df = pd.read_csv('diversion_diversion_or_energy_shortage_ensemble_00.csv', parse_dates=['time'])
diversion_target_df = pd.read_csv('diversion_diversion_or_energy_target_ensemble_00.csv', parse_dates=['time'])
shortage_ratio_df = pd.read_csv('diversion_shortage_ratio_ensemble_00.csv', parse_dates=['time'])

# Load reservoir data for ensemble 0
reservoir_elevation_df = pd.read_csv('reservoir_reservoir_water_surface_elevation_ensemble_00.csv', parse_dates=['time'])
reservoir_capacity_df = pd.read_csv('reservoir_reservoir_storage_capacity_ensemble_00.csv', parse_dates=['time'])
energy_generated_df = pd.read_csv('reservoir_energy_generated_ensemble_00.csv', parse_dates=['time'])
```

## Scripts
`explore_netcdf.py` walks through the opening of the NetCDF file and exploration of its contents.

`netcdf_to_csvs.py` contains the code used to produce the csv file outputs from the NetCDF.
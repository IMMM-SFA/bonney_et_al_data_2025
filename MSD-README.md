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
- **`right_id`**: Water right identifiers for shortage data

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
└── hmm_parameters/
    ├── hmm_parameters_ensemble_00.csv
    ├── hmm_parameters_ensemble_01.csv
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
```

## Scripts
`explore_netcdf.py` walks through the opening of the NetCDF file and exploration of its contents.

`netcdf_to_csvs.py` contains the code used to produce the csv file outputs from the NetCDF.
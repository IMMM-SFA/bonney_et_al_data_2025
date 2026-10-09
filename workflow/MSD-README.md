# Synthetic streamflow and WRAP water management outputs for three Texas river basins

This archive contains an ensemble of synthetic monthly streamflow realizations for the Colorado, Sabine and Trinity river basins in Texas, each paired with the water management outputs obtained by running that realization through the basin's Water Availability Model (WAM) in the Water Rights Analysis Package (WRAP). The code that produced it is at https://github.com/IMMM-SFA/bonney_et_al_data_2025.

## What is in the dataset

| Basin    | Outlet control point | Realizations | Months per realization | Period covered | Streamflow gages | Water rights | Reservoirs |
|----------|----------------------|--------------|------------------------|----------------|------------------|--------------|------------|
| Colorado | INK20000             | 2,000        | 924                    | 1940 to 2016   | 45               | 2,219        | 527        |
| Sabine   | IN  SRSL             | 2,000        | 708                    | 1940 to 1998   | 27               | 394          | 216        |
| Trinity  | IN 8TRGB             | 2,000        | 684                    | 1940 to 1996   | 40               | 1,072        | 699        |

Each realization spans the same months as the basin's historical WAM record. The realizations are not forecasts for those calendar years: the dates are the historical record's time axis, reused so that every realization is directly comparable with the historical simulation. The WAM is based in historical operation while the streamflows that are used as input are based on climate models for the 2020-2059 time period.

This repo contains one NetCDF dataset per basin, generated from the 9505 ensemble described below.

## How the dataset was generated

1. **Training data.** The DOE 9505 streamflow projection ensemble ([HydroSource](https://hydrosource.ornl.gov/data/datasets/9505v3_1/)) provides monthly flow on NHD river reaches. For each basin, the reach nearest the WAM outlet control point was identified, and the 2020 to 2059 projections of 48 ensemble members were extracted: six climate models (ACCESS-CM2, BCC-CSM2-MR, CNRM-ESM2-1, MPI-ESM1-2-HR, MRI-ESM2-0, NorESM2-MM) under SSP5-8.5, each downscaled with two methods (DBCCA, RegCM), bias-corrected to two datasets (Daymet, Livneh), and run through two hydrologic models (PRMS, VIC5).
2. **Annual model.** A two-state Bayesian hidden Markov model with log-normal emissions was fitted to the annual outlet flow of those members, with priors derived from the basin's historical record.
3. **Annual realizations.** For each realization a parameter set was drawn from the posterior and an annual outlet trajectory sampled from it. The pooled annual trajectories were then bias-corrected to the historical annual record by empirical quantile mapping, after multiplying the three lowest and three highest historical years by 0.8 and 1.2 respectively to widen the tails.
4. **Monthly disaggregation.** For each synthetic year an analog year was drawn from the historical record, with years of similar annual outlet flow more likely to be chosen. The outlet's monthly pattern from that analog year was rescaled to the synthetic annual total, and every other control point was set from its ratio to the outlet in the analog year. A small number of control points that lie outside the 9505 model domain or are accounting placeholders (listed as `fixed_control_points` in `basins.json`) carry their historical values unchanged in every realization.
5. **WRAP simulation.** Each realization was written as a WRAP `.FLO` file together with a synthetic net-evaporation `.EVA` file, built by resampling historical net evaporation at each reservoir from years whose flow at that reservoir's anchor control point resembles the synthetic flow. WRAP was run with the basin's WAM, and the diversion and reservoir outputs were extracted.
6. **Packaging.** Streamflow, hidden states, model parameters, WRAP outputs and water-right metadata were combined into one NetCDF file per basin, converted to 32-bit floats and compressed into a single NetCDF per basin.

## Files

```
README.md                                    this file
Colorado/All Models_colorado_synthetic_dataset.nc
Sabine/All Models_sabine_synthetic_dataset.nc
Trinity/All Models_trinity_synthetic_dataset.nc
data/                                        inputs needed to reproduce the dataset (see below)
```

## NetCDF structure

### Dimensions
| Dimension            | Meaning                                                      |
|----------------------|--------------------------------------------------------------|
| `realization`        | Index of the synthetic realization, 0 to 1999                |
| `time_step`          | Monthly time steps (datetime64), first of each month         |
| `gage_id`            | WRAP control point identifiers for streamflow                |
| `year`               | Calendar years of the record, as strings, for the annual hidden states |
| `hmm_parameter_name` | Labels of the HMM parameters                                 |
| `right_id`           | WRAP water right identifiers                                 |
| `reservoir_id`       | WRAP reservoir identifiers                                   |

### Data variables
Units and descriptions are stored as attributes on each variable and mirror `data/configs/hmm_synthetic_data_metadata.json` and `data/configs/wrap_variable_metadata.json`.

| Variable | Dimensions | Units | Description |
|----------|------------|-------|-------------|
| `synthetic_streamflow` | realization, time_step, gage_id | acre-feet | Monthly synthetic streamflow at every control point. |
| `annual_wet_dry_state` | realization, year | 0 or 1 | The HMM state that emitted each year's annual outlet flow: 0 for the lower-mean state, 1 for the higher-mean state. |
| `hmm_parameters` | realization, hmm_parameter_name | varies | The posterior parameter draw used for the realization: state means and standard deviations in log space, the transition matrix and the initial state distribution. |
| `diversion_or_energy_shortage` | realization, time_step, right_id | acre-feet | WRAP shortage for each water right. |
| `diversion_or_energy_target` | realization, time_step, right_id | acre-feet | WRAP target for each water right. |
| `shortage_ratio` | realization, time_step, right_id | ratio | `1 - (target - shortage) / target`; 0 is no shortage, 1 is full shortage. Undefined (NaN) where the target is zero. |
| `reservoir_water_surface_elevation` | realization, time_step, reservoir_id | feet (unverified) | Water surface elevation. |
| `reservoir_storage_capacity` | realization, time_step, reservoir_id | acre-feet (unverified) | End-of-month storage. |
| `inflows_to_reservoir_from_stream_flow_depletions` | realization, time_step, reservoir_id | acre-feet (unverified) | Inflow from streamflow depletions. |
| `inflows_to_reservoir_from_releases_from_other_reservoirs` | realization, time_step, reservoir_id | acre-feet (unverified) | Inflow released from other reservoirs. |
| `reservoir_net_evaporation_precipitation_volume` | realization, time_step, reservoir_id | acre-feet (unverified) | Net evaporation minus precipitation volume. |
| `energy_generated` | realization, time_step, reservoir_id | MWh (unverified) | Hydroelectric energy generated. |
| `reservoir_releases_accessible_to_hydroelectric_power_turbines` | realization, time_step, reservoir_id | acre-feet (unverified) | Releases through turbines. |
| `reservoir_releases_not_accessible_to_hydroelectric_power_turbines` | realization, time_step, reservoir_id | acre-feet (unverified) | Releases bypassing turbines. |
| `sector_raw` | right_id | | The water right's `use` code verbatim from the WAM `.DAT` file. |
| `sector` | right_id | | The use code bucketed into IND, IRR, MIN, MUN, POW, REC or OTHER. |
| `priority_number` | right_id | | The water right's priority number verbatim from the WAM, usually a YYYYMMDD appropriation date. |
| `priority_date` | right_id | datetime64 | `priority_number` parsed as a date; null where the number is not a valid date. |

### Global attributes
`basin_name`, `wrap_outflow_gage`, `9505_reach_id`, `subset_name`, `ensemble_filters` (the 9505 subset definition), `n_realizations`, `n_months`, `n_years`, `n_gages`, `n_hmm_parameters`, `start_year`, `end_year`, `creation_date` (of the streamflow generation step), `generation_method`, `temporal_resolution`, `spatial_resolution`, `source`, `title`.

### Notes for users
- Bias correction is applied to the pooled ensemble, so the distribution of annual outlet flow across all realizations and years matches the historical record (with widened tails), while each realization's sequence of years is as generated by the HMM.
- A small number of water rights in each WAM are accounting placeholders with very large targets. Basin-wide sums of shortage or target volumes are dominated by them; `shortage_ratio` is bounded per right and can be a useful basis for basin-wide statistics.
- WRAP reports negative shortages for a few rights under its surplus convention. For those rights `shortage_ratio` falls outside 0 to 1, and it is infinite where the target is also zero.
- Some `reservoir_id` entries are WRAP accounting constructs rather than physical reservoirs and have no storage output (NaN). Water surface elevation and energy variables are zero in basins whose WAM does not define elevation tables or hydropower rights.
- `priority_date` is null for rights whose priority number is a sentinel or malformed.

## `data` folder

Everything needed to rerun the workflow. Paths below are relative to `data/`.

```
configs/
  basins.json                       basin definitions (see below)
  ensemble_filters.json             the 9505 subsets, as filters on member attributes
  random_seeds.json                 seeds for every random step of the workflow
  hmm_synthetic_data_metadata.json  attribute metadata for the streamflow and HMM variables
  wrap_variable_metadata.json       attribute metadata for the WRAP and water-right variables
geospatial/
  9505_shapefiles/                  NHD flowline shapefiles for HUC2 regions 11, 12 and 13 (from the 9505 download)
  wrap_gages/                       primary control point locations for each basin
WRAP/
  basin_wams/                       the three WAMs: .DAT (two variants), .DIS, .EVA, .FLO and, for the Colorado, .FAD and .HIS
  SIM.exe                           the WRAP simulation executable (Windows binary, run through Wine on Linux)
```

`basins.json` has one entry per basin with: `gage_name` (the outlet control point), `reach_id` (the 9505 reach associated with it), `flo_file` (path to the WAM's `.FLO`), `usgs_gage_id` (where available), `fixed_control_points` (control points carried unchanged from the historical record) and `reservoir_anchors` (for each reservoir evaporation site, the `anchor_cp` used for evaporation disaggregation and the `anchor_r_squared` of the fit that selected it).

The geospatial data are needed only for the first stage of the workflow, which maps control points to 9505 reaches. The shapefiles are a subset of the 9505 download: only HUC2 regions 11, 12 and 13 are included.

## Loading the data in Python

Requires `xarray` and `netCDF4` or `h5netcdf`.

```python
import xarray as xr

ds = xr.open_dataset("Colorado/All Models_colorado_synthetic_dataset.nc")

streamflow = ds["synthetic_streamflow"]            # (realization, time_step, gage_id)
outlet = streamflow.sel(gage_id=ds.attrs["wrap_outflow_gage"])
annual_outlet = outlet.groupby("time_step.year").sum()

shortage_ratio = ds["shortage_ratio"]             # (realization, time_step, right_id)
municipal = shortage_ratio.sel(right_id=ds["sector"] == "MUN")

storage = ds["reservoir_storage_capacity"]        # (realization, time_step, reservoir_id)
states = ds["annual_wet_dry_state"]               # (realization, year)
```

The three-dimensional WRAP variables are large (the Colorado shortage variables are 2,000 by 924 by 2,219). Select a subset of realizations or rights before calling `.values` or `.load()`.

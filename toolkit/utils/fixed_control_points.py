from typing import List, Tuple

import numpy as np
import pandas as pd


def get_fixed_control_points(basin_config: dict) -> List[str]:
    """CP ids to hold fixed for this basin, per basins.json's "fixed_control_points" list."""
    return list(basin_config.get("fixed_control_points", []))


def split_free_and_fixed(site_names: List[str], basin_config: dict) -> Tuple[List[str], List[str]]:
    """Partition site_names into (free, fixed), each preserving site_names' relative order."""
    fixed = set(get_fixed_control_points(basin_config))
    free_sites = [s for s in site_names if s not in fixed]
    fixed_sites = [s for s in site_names if s in fixed]
    return free_sites, fixed_sites


def splice_fixed_columns(
    generated_monthly: np.ndarray,
    free_sites: List[str],
    historical_monthly: pd.DataFrame,
    basin_config: dict,
    site_names: List[str],
) -> np.ndarray:
    """Reinsert fixed CPs' historical monthly values into generated output (shape
    (..., n_months, len(free_sites)), any number of leading dims), reordered to site_names.
    Requires n_months == len(historical_monthly).
    """
    fixed_sites = get_fixed_control_points(basin_config)
    n_months = generated_monthly.shape[-2]

    if fixed_sites and n_months != len(historical_monthly):
        raise ValueError(
            f"Basin has {len(fixed_sites)} fixed_control_points, which require the synthetic "
            f"horizon to match the historical record exactly ({len(historical_monthly)} "
            f"months), got {n_months} months instead."
        )

    free_pos = {site: i for i, site in enumerate(free_sites)}
    out_shape = generated_monthly.shape[:-1] + (len(site_names),)
    out = np.empty(out_shape, dtype=generated_monthly.dtype)

    for j, site in enumerate(site_names):
        if site in free_pos:
            out[..., j] = generated_monthly[..., free_pos[site]]
        else:
            out[..., j] = historical_monthly[site].astype(float).to_numpy()

    return out


def overwrite_fixed_columns(
    synth_flow: pd.DataFrame,
    historical_monthly: pd.DataFrame,
    basin_config: dict,
) -> pd.DataFrame:
    """Overwrite fixed_control_points columns of an already-generated, full-width synthetic
    DataFrame with their historical monthly values, positionally (independent of either
    frame's actual date index). Requires len(synth_flow) == len(historical_monthly).
    """
    fixed_sites = get_fixed_control_points(basin_config)
    if fixed_sites and len(synth_flow) != len(historical_monthly):
        raise ValueError(
            f"Basin has {len(fixed_sites)} fixed_control_points, which require synth_flow to "
            f"match the historical record's length exactly ({len(historical_monthly)} rows), "
            f"got {len(synth_flow)} rows instead."
        )

    for cp in fixed_sites:
        synth_flow[cp] = historical_monthly[cp].astype(float).to_numpy()
    return synth_flow

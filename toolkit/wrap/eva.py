from typing import Optional

import numpy as np
import pandas as pd

from toolkit.utils.disaggregation import Disaggregator


def generate_eva_df(
    synthetic_flow_df: pd.DataFrame,
    historical_eva_df: pd.DataFrame,
    historical_flow_df: pd.DataFrame,
    basin_config: dict,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Generate a synthetic per-realization .EVA-compatible DataFrame.

    Parameters
    ----------
    synthetic_flow_df : DataFrame
        Shape (n_months, n_cps), columns = CP IDs, DatetimeIndex, full calendar years.
    historical_eva_df : DataFrame
        Output of evp_to_df. Shape (n_hist_months, n_sites).
    historical_flow_df : DataFrame
        Output of flo_to_df. Shape (n_hist_months, n_cps).
    basin_config : dict
        basins.json entry for the basin, including "reservoir_anchors":
        {eva_site: {"anchor_cp": cp_id, "anchor_r_squared": r2}}.
    rng : np.random.Generator, optional
        Random number generator for analog-year sampling. If None, a fresh
        default_rng() is used.

    Returns
    -------
    DataFrame
        Same format as evp_to_df output (DatetimeIndex, one column per EVA site),
        compatible with df_to_evp.
    """
    if rng is None:
        rng = np.random.default_rng()

    historical_eva_df = historical_eva_df.astype(float)
    historical_flow_df = historical_flow_df.astype(float)
    reservoir_anchors = basin_config.get("reservoir_anchors", {})

    eva_sites = [c for c in historical_eva_df.columns if isinstance(c, str) and c.startswith("EV")]
    synthetic_flow_annual = synthetic_flow_df.groupby(synthetic_flow_df.index.year).sum()

    synthetic_eva = pd.DataFrame(index=synthetic_flow_df.index, columns=eva_sites, dtype=float)

    for site in eva_sites:
        hist_eva_site = historical_eva_df[site]
        anchor = reservoir_anchors.get(site)

        if anchor is not None:
            anchor_cp = anchor["anchor_cp"]
            disaggregator = Disaggregator(historical_flow_df[anchor_cp], rng=rng)
            analog_years = disaggregator.select_analog_years(synthetic_flow_annual[anchor_cp])
            series = disaggregator.stamp_verbatim(hist_eva_site, analog_years, synthetic_flow_df.index)
        else:
            raise Exception

        synthetic_eva[site] = series.to_numpy()

    return synthetic_eva

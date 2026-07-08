"""Generate per-realization synthetic .EVA data that co-varies with synthetic streamflow.

See kirklocal/plan_reservoir_eva.md (Stage 3) and kirklocal/eva_update.md for background.
Built on `toolkit.utils.disaggregation.Disaggregator`, the same analog-year "stamp"
machinery behind `toolkit.hmm.disaggregation.disaggregate_annual_to_monthly` (the actual
flow_h -> flow_s pipeline): for each synthetic year, find the historical year whose
annual flow at the reservoir's anchor control point is closest (k-nearest-neighbor,
inverse-rank-weighted selection), and reuse -- verbatim, unscaled -- that historical
year's actual monthly net evap values (`Disaggregator.stamp_verbatim`).

Net evap is *not* rescaled by a flow ratio the way flow sites are rescaled relative to
their anchor (`Disaggregator.stamp_ratio`): that scaling assumes the target quantity
moves in the same direction as the anchor's flow, which is true for flow-to-flow (both
are basin "wetness") but backwards for flow-to-evap (net evap and flow are negatively
correlated -- see Stage 1's correlation analysis). Verbatim reuse sidesteps that sign
mismatch entirely, relying on the neighbor search to find a historically similar year.

Each EVA site draws its own independent analog year, even when two sites share the same
anchor CP -- reservoirs are not locked together just because they happen to correlate
best with the same control point. (This is a deliberate choice, revisit if it turns out
reservoirs sharing an anchor should be forced into the same analog year instead.)

Sites without a strong anchor (excluded from basins.json's "reservoir_anchors") fall back
to their historical monthly climatology, unchanged across realizations.
"""

from typing import Optional

import numpy as np
import pandas as pd

from toolkit.utils.disaggregation import Disaggregator


def _monthly_climatology(hist_eva: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    climatology = hist_eva.groupby(hist_eva.index.month).mean()
    values = climatology.loc[target_index.month].to_numpy()
    return pd.Series(values, index=target_index)


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
        {eva_site: {"anchor_cp": cp_id, "anchor_correlation": r}}.
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
            # A fresh Disaggregator (and analog-year draw) per site, even when sites
            # share an anchor CP -- see module docstring.
            disaggregator = Disaggregator(historical_flow_df[anchor_cp], rng=rng)
            analog_years = disaggregator.select_analog_years(synthetic_flow_annual[anchor_cp])
            series = disaggregator.stamp_verbatim(hist_eva_site, analog_years, synthetic_flow_df.index)
        else:
            series = _monthly_climatology(hist_eva_site, synthetic_flow_df.index)

        synthetic_eva[site] = series.to_numpy()

    return synthetic_eva

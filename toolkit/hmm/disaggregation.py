from typing import Optional

import numpy as np
import pandas as pd

from toolkit.utils.disaggregation import Disaggregator


def disaggregate_annual_to_monthly(
    annual_values: np.ndarray,
    historical_monthly_data: np.ndarray,
    anchor_index: int,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Disaggregate annual values to monthly values using historical spatio-temporal patterns.

    For each synthetic annual value, finds the k nearest historical years by distance to the
    anchor site's historical annual total, samples one neighbor (inverse-rank weighted), and
    uses that neighbor's monthly pattern to disaggregate the anchor site temporally and every
    other site spatially (relative to the anchor).

    The anchor site does not need to be an outlet/outflow gage - any site whose annual pattern
    is a representative proxy for the others can be used (e.g. a reservoir's correlated control
    point).

    Thin numpy-array-in/numpy-array-out wrapper around `toolkit.utils.disaggregation.Disaggregator`:
    the anchor's own temporal disaggregation is `stamp_temporal_rescale`, and every site
    (including the anchor itself, whose ratio-to-itself is exactly 1) is disaggregated via
    `stamp_ratio` against the anchor's already-computed synthetic monthly values.

    Parameters
    ----------
    annual_values : np.ndarray
        Annual values to disaggregate, shape (num_years,)
    historical_monthly_data : np.ndarray
        Historical monthly data with shape (hist_years * 12, n_sites)
    anchor_index : int
        Column index of the anchor site used for neighbor distance and temporal disaggregation
    rng : Optional[np.random.Generator], default=None
        Random number generator for neighbor sampling. If None, a fresh default_rng() is used.

    Returns
    -------
    pd.DataFrame
        Disaggregated monthly data with shape (num_years * 12, n_sites)
    """
    if rng is None:
        rng = np.random.default_rng()

    num_years = len(annual_values)
    hist_years = int(historical_monthly_data.shape[0] / 12)
    num_sites = historical_monthly_data.shape[1]

    site_names = [f"site_{i}" for i in range(num_sites)]
    hist_index = pd.date_range("1900-01", periods=hist_years * 12, freq="MS")
    historical_df = pd.DataFrame(historical_monthly_data, index=hist_index, columns=site_names)

    synth_years = np.arange(2000, 2000 + num_years)
    target_index = pd.date_range("2000-01", periods=num_years * 12, freq="MS")
    driver_synthetic_annual = pd.Series(annual_values, index=synth_years)

    disaggregator = Disaggregator(historical_df[site_names[anchor_index]], rng=rng)
    analog_years = disaggregator.select_analog_years(driver_synthetic_annual)

    anchor_synthetic_monthly = disaggregator.stamp_temporal_rescale(
        analog_years, driver_synthetic_annual, target_index
    )
    return disaggregator.stamp_ratio(historical_df, analog_years, anchor_synthetic_monthly)

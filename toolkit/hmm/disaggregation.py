from typing import Optional

import numpy as np
import pandas as pd


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
    hist_monthly = historical_monthly_data.reshape(hist_years, 12, -1)
    num_sites = hist_monthly.shape[2]

    # Anchor site historical data
    anchor_hist_monthly = hist_monthly[:, :, anchor_index]
    anchor_hist_annual = np.sum(anchor_hist_monthly, axis=1)

    # Distance between synthetic and historical annual values at the anchor site
    annual_distances = np.abs(np.subtract.outer(annual_values, anchor_hist_annual))

    # Spatial ratios: each site relative to the anchor site
    site_ratios = np.zeros(hist_monthly.shape)
    for i in range(num_sites):
        site_ratios[:, :, i] = hist_monthly[:, :, i] / anchor_hist_monthly

    # Temporal breakdown: anchor monthly relative to anchor annual
    anchor_temporal_breakdown = np.zeros((hist_years, 12))
    for i in range(hist_years):
        anchor_temporal_breakdown[i, :] = anchor_hist_monthly[i, :] / anchor_hist_annual[i]

    # Neighbor selection probabilities (inverse rank weighting)
    k = int(np.sqrt(hist_years))
    neighbor_probabilities = np.array([1 / (j + 1) for j in range(k)])
    neighbor_probabilities = neighbor_probabilities / np.sum(neighbor_probabilities)

    synth_monthly = np.zeros((num_years, 12, num_sites))
    for j in range(num_years):
        # k nearest neighbors by anchor annual distance
        indices = np.argsort(annual_distances[j])[:k]
        neighbor_idx = rng.choice(indices, p=neighbor_probabilities)

        # Disaggregate anchor site temporally
        synth_monthly[j, :, anchor_index] = (
            anchor_temporal_breakdown[neighbor_idx, :] * annual_values[j]
        )

        # Disaggregate all sites spatially relative to the anchor
        for n in range(12):
            synth_monthly[j, n, :] = (
                site_ratios[neighbor_idx, n, :] * synth_monthly[j, n, anchor_index]
            )

    synth_monthly = synth_monthly.reshape(num_years * 12, num_sites)

    start = "2000-01"
    end = str(2000 + num_years - 1) + "-12"
    time_range = pd.date_range(start=start, end=end, freq="MS")

    return pd.DataFrame(
        synth_monthly,
        index=time_range,
        columns=[f"site_{i}" for i in range(num_sites)],
    )

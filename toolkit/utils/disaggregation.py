from typing import Optional, Union

import numpy as np
import pandas as pd


def select_analog_year_indices(
    synthetic_annual_values: np.ndarray,
    historical_annual_values: np.ndarray,
    rng: np.random.Generator,
    weighting: str = "inverse_rank",
) -> np.ndarray:
    """For each synthetic annual value, sample one analog historical year.

    Finds the k = sqrt(n_hist_years) nearest historical years by absolute distance to
    the synthetic annual value, then samples one via `weighting`:
    - "inverse_rank": closest neighbor most likely, weights 1/1, 1/2, ..., 1/k.
    - "uniform": every one of the k neighbors equally likely.

    Parameters
    ----------
    synthetic_annual_values : np.ndarray
        Shape (num_years,).
    historical_annual_values : np.ndarray
        Shape (hist_years,).
    rng : np.random.Generator
    weighting : str, default="inverse_rank"
        "inverse_rank" or "uniform".

    Returns
    -------
    np.ndarray
        Shape (num_years,), each entry a positional index into historical_annual_values.
    """
    hist_years = len(historical_annual_values)
    k = int(np.sqrt(hist_years))
    if weighting == "inverse_rank":
        neighbor_probabilities = np.array([1 / (j + 1) for j in range(k)])
        neighbor_probabilities = neighbor_probabilities / np.sum(neighbor_probabilities)
    elif weighting == "uniform":
        neighbor_probabilities = np.full(k, 1 / k)
    else:
        raise ValueError(f"Unknown weighting: {weighting!r}")

    distances = np.abs(np.subtract.outer(synthetic_annual_values, historical_annual_values))
    neighbor_indices = np.empty(len(synthetic_annual_values), dtype=int)
    for j in range(len(synthetic_annual_values)):
        candidates = np.argsort(distances[j])[:k]
        neighbor_indices[j] = rng.choice(candidates, p=neighbor_probabilities)
    return neighbor_indices


class Disaggregator:
    """Analog-year disaggregation built around one "driver" series' historical record.

    `select_analog_years` performs the k-nearest-neighbor matching once; its result
    (one historical calendar year per synthetic year) is then reused across one or more
    `stamp_*` calls, each expressing a different physical relationship between a target
    series and the driver:

    - `stamp_temporal_rescale`: disaggregate the driver's *own* synthetic annual totals
      to monthly, using each analog year's month/annual fraction of its own historical
      record. (There's no separate "target" here -- the driver is the target.)
    - `stamp_ratio`: scale a target by its analog year's ratio to the driver, applied
      to the driver's already-realized synthetic monthly values. Appropriate when
      target and driver are the same kind of quantity, expected to move in the same
      direction (e.g. flow at two control points in the same basin).
    - `stamp_verbatim`: reuse each analog year's actual monthly target values, with no
      rescaling at all. Appropriate when the target does *not* move proportionally
      with the driver -- e.g. a negatively-correlated quantity, where ratio-scaling
      would get the local response direction backwards.

    `target_historical` for any `stamp_*` call may be a single Series or a DataFrame of
    several target columns computed at once (vectorized); either way it must share the
    exact same historical monthly DatetimeIndex as the driver.

    `weighting` controls how `select_analog_years` samples among the k nearest neighbors
    -- see `select_analog_year_indices`. Default "inverse_rank" favors closer years;
    "uniform" gives every one of the k neighbors equal odds.
    """

    def __init__(
        self,
        driver_historical: pd.Series,
        rng: Optional[np.random.Generator] = None,
        weighting: str = "inverse_rank",
    ):
        self.driver_historical = driver_historical.astype(float)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.weighting = weighting

        self._hist_years = np.sort(self.driver_historical.index.year.unique())
        self._year_to_pos = {year: pos for pos, year in enumerate(self._hist_years)}
        self._driver_monthly = self.driver_historical.to_numpy().reshape(len(self._hist_years), 12)
        self._driver_annual = self._driver_monthly.sum(axis=1)

    def select_analog_years(self, driver_synthetic_annual: pd.Series) -> pd.Series:
        """For each synthetic year (index of driver_synthetic_annual), pick one analog
        historical calendar year.

        Returns a Series indexed the same as driver_synthetic_annual, valued with the
        chosen historical calendar year (int). Pass the result to one or more
        `stamp_*` calls.
        """
        positions = select_analog_year_indices(
            driver_synthetic_annual.to_numpy(),
            self._driver_annual,
            self.rng,
            weighting=self.weighting,
        )
        neighbor_years = self._hist_years[positions]
        return pd.Series(neighbor_years, index=driver_synthetic_annual.index, name="analog_year")

    def stamp_temporal_rescale(
        self,
        analog_years: pd.Series,
        driver_synthetic_annual: pd.Series,
        target_index: pd.DatetimeIndex,
    ) -> pd.Series:
        neighbor_positions = self._neighbor_positions(analog_years)
        fractions = self._driver_monthly[neighbor_positions, :] / self._driver_annual[neighbor_positions, None]
        values = fractions * driver_synthetic_annual.to_numpy()[:, None]
        return pd.Series(values.reshape(-1), index=target_index, name=self.driver_historical.name)

    def stamp_ratio(
        self,
        target_historical: Union[pd.Series, pd.DataFrame],
        analog_years: pd.Series,
        driver_synthetic_monthly: pd.Series,
    ) -> Union[pd.Series, pd.DataFrame]:
        neighbor_positions = self._neighbor_positions(analog_years)
        target_monthly, is_series, columns = self._reshape_target(target_historical)

        ratio = target_monthly[neighbor_positions, :, :] / self._driver_monthly[neighbor_positions, :, None]
        n_synth_years = len(analog_years)
        scale_monthly = driver_synthetic_monthly.to_numpy().reshape(n_synth_years, 12)
        out = ratio * scale_monthly[:, :, None]

        return self._to_output(out, driver_synthetic_monthly.index, columns, is_series)

    def stamp_verbatim(
        self,
        target_historical: Union[pd.Series, pd.DataFrame],
        analog_years: pd.Series,
        target_index: pd.DatetimeIndex,
    ) -> Union[pd.Series, pd.DataFrame]:
        neighbor_positions = self._neighbor_positions(analog_years)
        target_monthly, is_series, columns = self._reshape_target(target_historical)

        out = target_monthly[neighbor_positions, :, :]

        return self._to_output(out, target_index, columns, is_series)

    def _neighbor_positions(self, analog_years: pd.Series) -> np.ndarray:
        return np.array([self._year_to_pos[year] for year in analog_years.to_numpy()])

    def _reshape_target(self, target_historical):
        is_series = isinstance(target_historical, pd.Series)
        target_df = target_historical.to_frame() if is_series else target_historical
        n_targets = target_df.shape[1]
        target_monthly = target_df.to_numpy().reshape(len(self._hist_years), 12, n_targets)
        return target_monthly, is_series, target_df.columns

    @staticmethod
    def _to_output(values: np.ndarray, target_index, columns, is_series: bool):
        n_synth_years, _, n_targets = values.shape
        flat = values.reshape(n_synth_years * 12, n_targets)
        result = pd.DataFrame(flat, index=target_index, columns=columns)
        return result.iloc[:, 0].rename(columns[0]) if is_series else result

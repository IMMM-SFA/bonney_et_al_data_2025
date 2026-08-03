import numpy as np
import pandas as pd


def driest_n_year_mean(annual: np.ndarray, n: int) -> float:
    """Lowest n-year rolling mean in the series (n=1 is just the single driest year)."""
    annual = np.asarray(annual, dtype=float)
    if n == 1:
        return float(annual.min())
    rolling = np.convolve(annual, np.ones(n) / n, mode="valid")
    return float(rolling.min())


def flashiness(annual: np.ndarray) -> int:
    """Count of transitions between the series' driest and wettest quartiles."""
    annual = np.asarray(annual, dtype=float)
    q25, q75 = np.percentile(annual, [25, 75])
    state = np.where(annual <= q25, -1, np.where(annual >= q75, 1, 0))
    transitions = 0
    for prev, curr in zip(state[:-1], state[1:]):
        if (prev == -1 and curr == 1) or (prev == 1 and curr == -1):
            transitions += 1
    return transitions


def drought_duration_stats(annual: np.ndarray, drought_threshold: float) -> tuple:
    """(avg_duration, max_duration) of consecutive runs at or below `drought_threshold`."""
    annual = np.asarray(annual, dtype=float)
    durations = []
    count = 0
    for value in annual:
        if value <= drought_threshold:
            count += 1
        elif count > 0:
            durations.append(count)
            count = 0
    if count > 0:
        durations.append(count)
    if not durations:
        return 0.0, 0
    return float(np.mean(durations)), int(np.max(durations))


def decadal_variability(annual: np.ndarray, window: int = 10) -> float:
    """Std deviation of rolling `window`-year means."""
    annual = np.asarray(annual, dtype=float)
    rolling = np.convolve(annual, np.ones(window) / window, mode="valid")
    return float(np.std(rolling))


def compute_drought_metrics(annual: np.ndarray, drought_threshold: float) -> dict:
    """All metrics above for a single annual series, plus mean/median."""
    annual = np.asarray(annual, dtype=float)
    avg_duration, max_duration = drought_duration_stats(annual, drought_threshold)
    return {
        "mean": float(annual.mean()),
        "median": float(np.median(annual)),
        "driest_1": driest_n_year_mean(annual, 1),
        "driest_3": driest_n_year_mean(annual, 3),
        "driest_5": driest_n_year_mean(annual, 5),
        "driest_10": driest_n_year_mean(annual, 10),
        "flashiness": flashiness(annual),
        "avg_drought_duration": avg_duration,
        "max_drought_duration": max_duration,
        "decadal_variability": decadal_variability(annual),
    }


def compute_drought_metrics_ensemble(annual_ensemble: np.ndarray, historical_annual: np.ndarray):
    historical_annual = np.asarray(historical_annual, dtype=float)
    drought_threshold = np.percentile(historical_annual, 25)

    historical_metrics = compute_drought_metrics(historical_annual, drought_threshold)
    metrics_df = pd.DataFrame(
        compute_drought_metrics(realization, drought_threshold) for realization in annual_ensemble
    )
    return metrics_df, historical_metrics

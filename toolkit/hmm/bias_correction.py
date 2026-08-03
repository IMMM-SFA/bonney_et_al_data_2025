from typing import Optional

import numpy as np
from scipy.stats import rankdata


def delta_scaling(hist_annual: np.ndarray, synth_annual: np.ndarray) -> np.ndarray:
    """Rescale the whole ensemble by a single factor so its grand mean matches history."""
    delta_factor = np.mean(hist_annual) / np.mean(synth_annual)
    return synth_annual * delta_factor


def variance_scaling(hist_annual: np.ndarray, synth_annual: np.ndarray) -> np.ndarray:
    """Rescale ensemble anomalies around the ensemble mean to match historical variance."""
    mu_hist = np.mean(hist_annual)
    mu_ens = np.mean(synth_annual)
    sigma_hist = np.std(hist_annual, ddof=1)
    sigma_ens = np.std(synth_annual, ddof=1)

    variance_ratio = sigma_hist / sigma_ens
    anomalies = synth_annual - mu_ens
    dry_variance_ratio = min(variance_ratio, 0.97 * (mu_hist / abs(anomalies.min())))

    corrected = np.where(
        anomalies >= 0,
        mu_hist + anomalies * variance_ratio,
        mu_hist + anomalies * dry_variance_ratio,
    )
    return np.maximum(0, corrected)


def log_variance_scaling(hist_annual: np.ndarray, synth_annual: np.ndarray) -> np.ndarray:
    """Like `variance_scaling`, but matches variance in log space (symmetric, no dry/wet split)."""
    log_hist = np.log(hist_annual)
    log_synth = np.log(synth_annual)

    ratio = np.std(log_hist, ddof=1) / np.std(log_synth, ddof=1)
    corrected_log = np.mean(log_hist) + (log_synth - np.mean(log_synth)) * ratio
    return np.exp(corrected_log)


def empirical_quantile_mapping(hist_annual: np.ndarray, synth_annual: np.ndarray) -> np.ndarray:
    """Map the whole ensemble's CDF onto the historical CDF, lumped across realizations."""
    synth_annual = np.asarray(synth_annual)
    flat_synth = synth_annual.flatten()

    ranks = np.percentile(flat_synth, np.arange(101))
    synth_percentiles = np.interp(flat_synth, ranks, np.arange(101))
    hist_percentiles = np.percentile(hist_annual, synth_percentiles)

    return hist_percentiles.reshape(synth_annual.shape)


def individual_quantile_mapping(hist_annual: np.ndarray, synth_annual: np.ndarray) -> np.ndarray:
    """Map each realization's own rank onto the historical sorted values, preserving order."""
    synth_annual = np.asarray(synth_annual)
    sorted_hist = np.sort(np.asarray(hist_annual, dtype=float))
    percentile_grid = np.linspace(0, 100, len(sorted_hist))

    corrected = np.empty_like(synth_annual, dtype=float)
    for i in range(synth_annual.shape[0]):
        trajectory = synth_annual[i, :]
        ranks = (rankdata(trajectory, method="average") - 1) / (len(trajectory) - 1) * 100
        corrected[i, :] = np.interp(ranks, percentile_grid, sorted_hist)
    return corrected


def _stretch_historical_tails(hist_annual: np.ndarray, factor: float, tail_method: str) -> np.ndarray:
    """Exaggerate the historical record's tails, for wider extremes than observed alone."""
    sorted_hist = np.sort(np.asarray(hist_annual, dtype=float))
    if tail_method == "A":
        sorted_hist[0] *= (1.0 - factor)
    elif tail_method == "B":
        sorted_hist[:3] *= (1.0 - factor)
    elif tail_method == "C":
        sorted_hist[:3] *= (1.0 - factor)
        sorted_hist[-3:] *= (1.0 + factor)
    else:
        raise ValueError(f"Unknown tail_method: {tail_method!r}. Choose 'A', 'B', or 'C'.")
    return sorted_hist


def stretched_quantile_mapping(
    hist_annual: np.ndarray,
    synth_annual: np.ndarray,
    factor: float = 0.2,
    tail_method: str = "C",
) -> np.ndarray:
    """Empirical quantile mapping against a tail-stretched historical record."""
    stretched_hist = _stretch_historical_tails(hist_annual, factor=factor, tail_method=tail_method)
    return empirical_quantile_mapping(stretched_hist, synth_annual)


_METHODS = {
    "delta": delta_scaling,
    "variance": variance_scaling,
    "log_variance": log_variance_scaling,
    "empirical_quantile": empirical_quantile_mapping,
    "individual_quantile": individual_quantile_mapping,
    "stretched_quantile": stretched_quantile_mapping,
}


def apply_bias_correction(
    method: Optional[str],
    hist_annual: np.ndarray,
    synth_annual: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Dispatch to a named bias correction method; `method=None` is a no-op passthrough."""
    if method is None:
        return synth_annual
    if method not in _METHODS:
        raise ValueError(f"Unknown bias correction method: {method!r}. Choose one of {list(_METHODS)} or None.")
    return _METHODS[method](hist_annual, synth_annual, **kwargs)

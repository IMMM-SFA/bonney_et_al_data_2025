import numpy as np

# Stretch factor/method for the historical record's tails, as used in the colleague-shared
# reference script that identified this as the best-performing bias correction method.
STRETCH_FACTOR = 0.2
STRETCH_TAIL_METHOD = "C"  # stretch both the bottom 3 and top 3 sorted historical years


def _stretch_historical_tails(hist_annual: np.ndarray) -> np.ndarray:
    """Exaggerate the historical record's tails, for wider extremes than observed alone."""
    sorted_hist = np.sort(np.asarray(hist_annual, dtype=float))
    sorted_hist[:3] *= (1.0 - STRETCH_FACTOR)
    sorted_hist[-3:] *= (1.0 + STRETCH_FACTOR)
    return sorted_hist


def apply_bias_correction(hist_annual: np.ndarray, synth_annual: np.ndarray) -> np.ndarray:
    """Empirical quantile mapping against a tail-stretched historical record -- the method
    flagged as best in the colleague-shared reference script."""
    stretched_hist = _stretch_historical_tails(hist_annual)

    synth_annual = np.asarray(synth_annual)
    flat_synth = synth_annual.flatten()
    ranks = np.percentile(flat_synth, np.arange(101))
    synth_percentiles = np.interp(flat_synth, ranks, np.arange(101))
    hist_percentiles = np.percentile(stretched_hist, synth_percentiles)

    return hist_percentiles.reshape(synth_annual.shape)

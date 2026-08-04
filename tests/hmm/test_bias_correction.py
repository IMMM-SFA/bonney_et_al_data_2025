"""
Unit tests for toolkit.hmm.bias_correction.apply_bias_correction: stretched-tail empirical
quantile mapping, applied to a synthetic annual ensemble (n_ensembles, num_years) against a
1-D historical annual array.
"""
import numpy as np
import pytest

from toolkit.hmm import bias_correction as bc


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_apply_bias_correction_preserves_shape(rng):
    hist = rng.uniform(500, 1500, size=60)
    synth = rng.uniform(0, 5000, size=(30, 40))

    corrected = bc.apply_bias_correction(hist, synth)

    assert corrected.shape == synth.shape


def test_apply_bias_correction_widens_range_beyond_raw_historical(rng):
    hist = rng.uniform(500, 1500, size=60)
    synth = rng.uniform(0, 5000, size=(30, 40))

    corrected = bc.apply_bias_correction(hist, synth)

    # Stretched tails push corrected extremes beyond the raw historical min/max.
    assert corrected.min() < hist.min()
    assert corrected.max() > hist.max()

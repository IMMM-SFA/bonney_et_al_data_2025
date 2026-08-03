"""
Unit tests for toolkit.hmm.bias_correction: each method operates on a synthetic annual
ensemble, shape (n_ensembles, num_years), against a 1-D historical annual array.
"""
import numpy as np
import pytest

from toolkit.hmm import bias_correction as bc


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_delta_scaling_matches_historical_mean(rng):
    hist = rng.normal(1000, 50, size=60)
    synth = rng.normal(500, 25, size=(20, 40))  # different mean entirely

    corrected = bc.delta_scaling(hist, synth)

    assert corrected.mean() == pytest.approx(hist.mean(), rel=1e-9)
    assert corrected.shape == synth.shape


def test_variance_scaling_matches_historical_variance_when_uncapped(rng):
    # Anomalies kept modest so the dry-side dampening cap never binds -- both branches use
    # the same ratio, so corrected std should closely match historical std.
    hist = rng.normal(1000, 200, size=60)
    synth = rng.normal(1000, 50, size=(50, 40))

    corrected = bc.variance_scaling(hist, synth)

    assert corrected.std(ddof=1) == pytest.approx(hist.std(ddof=1), rel=0.05)
    assert (corrected >= 0).all()


def test_variance_scaling_clips_negative_values():
    hist = np.array([100.0, 900.0, 1000.0, 1100.0, 1900.0])  # wide historical spread
    synth = np.full((5, 5), 1000.0)
    synth[0, 0] = 990.0  # small dry anomaly that would otherwise go negative when stretched

    corrected = bc.variance_scaling(hist, synth)

    assert (corrected >= 0).all()


def test_log_variance_scaling_is_always_positive(rng):
    hist = rng.uniform(500, 1500, size=60)
    synth = rng.uniform(100, 2000, size=(30, 40))

    corrected = bc.log_variance_scaling(hist, synth)

    assert (corrected > 0).all()
    assert corrected.shape == synth.shape


def test_empirical_quantile_mapping_stays_within_historical_range(rng):
    hist = rng.uniform(500, 1500, size=60)
    synth = rng.uniform(0, 5000, size=(30, 40))

    corrected = bc.empirical_quantile_mapping(hist, synth)

    assert corrected.min() >= hist.min() - 1e-6
    assert corrected.max() <= hist.max() + 1e-6
    assert corrected.shape == synth.shape


def test_individual_quantile_mapping_preserves_within_realization_order(rng):
    hist = rng.uniform(500, 1500, size=60)
    synth = rng.uniform(0, 5000, size=(10, 40))

    corrected = bc.individual_quantile_mapping(hist, synth)

    assert corrected.shape == synth.shape
    for row_orig, row_corrected in zip(synth, corrected):
        assert (np.argsort(row_orig) == np.argsort(row_corrected)).all()
    assert corrected.min() >= hist.min() - 1e-6
    assert corrected.max() <= hist.max() + 1e-6


def test_stretched_quantile_mapping_widens_range_beyond_plain_mapping(rng):
    hist = rng.uniform(500, 1500, size=60)
    synth = rng.uniform(0, 5000, size=(30, 40))

    plain = bc.empirical_quantile_mapping(hist, synth)
    stretched = bc.stretched_quantile_mapping(hist, synth, factor=0.2, tail_method="C")

    assert stretched.min() <= plain.min()
    assert stretched.max() >= plain.max()


def test_apply_bias_correction_none_is_passthrough(rng):
    synth = rng.uniform(0, 5000, size=(10, 40))
    hist = rng.uniform(500, 1500, size=60)

    result = bc.apply_bias_correction(None, hist, synth)

    assert result is synth


def test_apply_bias_correction_dispatches_to_named_method(rng):
    hist = rng.normal(1000, 50, size=60)
    synth = rng.normal(500, 25, size=(20, 40))

    dispatched = bc.apply_bias_correction("delta", hist, synth)
    direct = bc.delta_scaling(hist, synth)

    np.testing.assert_array_equal(dispatched, direct)


def test_apply_bias_correction_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown bias correction method"):
        bc.apply_bias_correction("not_a_method", np.array([1.0]), np.array([[1.0]]))

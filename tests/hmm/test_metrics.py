import numpy as np
import pytest

from toolkit.hmm import metrics


def test_driest_n_year_mean_n1_is_series_minimum():
    annual = np.array([10.0, 2.0, 8.0, 5.0])
    assert metrics.driest_n_year_mean(annual, 1) == 2.0


def test_driest_n_year_mean_rolling_window():
    # 3-year rolling means: [10,2,8]->6.67, [2,8,5]->5.0, [8,5,20]->11.0 -- driest is 5.0
    annual = np.array([10.0, 2.0, 8.0, 5.0, 20.0])
    assert metrics.driest_n_year_mean(annual, 3) == pytest.approx(5.0)


def test_flashiness_no_crossings_for_constant_series():
    annual = np.array([50.0] * 10)
    assert metrics.flashiness(annual) == 0


def test_flashiness_counts_low_high_transitions():
    # q25=1, q75=100 for this series; state sequence is [-1,-1,1,1,-1,1] -> 3 crossings.
    annual = np.array([1.0, 1.0, 100.0, 100.0, 1.0, 100.0])
    assert metrics.flashiness(annual) == 3


def test_drought_duration_stats_basic_runs():
    # Below threshold (<=5) at indices 0,1 (run of 2) and 4,5,6 (run of 3)
    annual = np.array([1.0, 2.0, 10.0, 10.0, 3.0, 4.0, 5.0, 10.0])
    avg_duration, max_duration = metrics.drought_duration_stats(annual, drought_threshold=5.0)
    assert max_duration == 3
    assert avg_duration == pytest.approx((2 + 3) / 2)


def test_drought_duration_stats_no_droughts():
    annual = np.array([10.0, 20.0, 30.0])
    avg_duration, max_duration = metrics.drought_duration_stats(annual, drought_threshold=5.0)
    assert avg_duration == 0.0
    assert max_duration == 0


def test_drought_duration_stats_trailing_run_included():
    annual = np.array([10.0, 1.0, 1.0])
    avg_duration, max_duration = metrics.drought_duration_stats(annual, drought_threshold=5.0)
    assert max_duration == 2
    assert avg_duration == pytest.approx(2.0)


def test_decadal_variability_zero_for_constant_series():
    annual = np.array([100.0] * 15)
    assert metrics.decadal_variability(annual, window=10) == pytest.approx(0.0)


def test_decadal_variability_positive_for_varying_decades():
    annual = np.concatenate([np.full(10, 50.0), np.full(10, 150.0)])
    assert metrics.decadal_variability(annual, window=10) > 0


def test_compute_drought_metrics_returns_all_expected_keys():
    annual = np.random.default_rng(0).uniform(100, 1000, size=30)
    result = metrics.compute_drought_metrics(annual, drought_threshold=200.0)
    expected_keys = {
        "mean", "median", "driest_1", "driest_3", "driest_5", "driest_10",
        "flashiness", "avg_drought_duration", "max_drought_duration", "decadal_variability",
    }
    assert set(result.keys()) == expected_keys


def test_compute_drought_metrics_ensemble_shapes_and_threshold():
    rng = np.random.default_rng(1)
    historical_annual = rng.uniform(100, 1000, size=40)
    annual_ensemble = rng.uniform(100, 1000, size=(5, 40))

    metrics_df, historical_metrics = metrics.compute_drought_metrics_ensemble(
        annual_ensemble, historical_annual
    )

    assert len(metrics_df) == 5
    assert set(metrics_df.columns) == set(historical_metrics.keys())
    assert historical_metrics["mean"] == pytest.approx(historical_annual.mean())

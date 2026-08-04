"""
Tests for toolkit.utils.fixed_control_points -- the single source of truth for CPs held
fixed at their historical values across every synthetic realization (out-of-basin gages,
non-hydrologic placeholder CPs), driven by basins.json's "fixed_control_points".
"""
import numpy as np
import pandas as pd
import pytest

from toolkit.utils.fixed_control_points import (
    get_fixed_control_points,
    overwrite_fixed_columns,
    splice_fixed_columns,
    split_free_and_fixed,
)


def _basin_config(fixed_sites):
    return {"fixed_control_points": list(fixed_sites)}


def test_get_fixed_control_points_empty_when_absent():
    assert get_fixed_control_points({}) == []


def test_get_fixed_control_points_reads_basin_config():
    config = _basin_config(["A", "B"])
    assert get_fixed_control_points(config) == ["A", "B"]


def test_split_free_and_fixed_preserves_order():
    config = _basin_config(["B", "D"])
    free, fixed = split_free_and_fixed(["A", "B", "C", "D"], config)
    assert free == ["A", "C"]
    assert fixed == ["B", "D"]


def test_split_free_and_fixed_no_fixed_sites():
    free, fixed = split_free_and_fixed(["A", "B"], {})
    assert free == ["A", "B"]
    assert fixed == []


def _historical_df(n_months, columns):
    index = pd.date_range("1940-01", periods=n_months, freq="MS")
    data = {col: np.arange(n_months) + i * 1000 for i, col in enumerate(columns)}
    return pd.DataFrame(data, index=index)


def test_splice_fixed_columns_reinserts_historical_values():
    config = _basin_config(["fixed_a"])
    site_names = ["free_a", "fixed_a", "free_b"]
    free_sites = ["free_a", "free_b"]
    historical = _historical_df(12, site_names)

    generated = np.zeros((2, 12, 2))  # (n_ensembles, n_months, len(free_sites))
    generated[..., 0] = 7.0  # free_a
    generated[..., 1] = 8.0  # free_b

    result = splice_fixed_columns(generated, free_sites, historical, config, site_names)

    assert result.shape == (2, 12, 3)
    np.testing.assert_allclose(result[:, :, site_names.index("free_a")], 7.0)
    np.testing.assert_allclose(result[:, :, site_names.index("free_b")], 8.0)
    for ens in range(2):
        np.testing.assert_allclose(
            result[ens, :, site_names.index("fixed_a")], historical["fixed_a"].to_numpy()
        )


def test_splice_fixed_columns_no_fixed_sites_is_passthrough_reordering():
    config = _basin_config([])
    site_names = ["a", "b"]
    historical = _historical_df(6, site_names)
    generated = np.random.default_rng(0).uniform(size=(3, 6, 2))

    result = splice_fixed_columns(generated, ["a", "b"], historical, config, site_names)

    np.testing.assert_array_equal(result, generated)


def test_splice_fixed_columns_raises_on_length_mismatch():
    config = _basin_config(["fixed_a"])
    site_names = ["free_a", "fixed_a"]
    historical = _historical_df(12, site_names)
    generated = np.zeros((1, 6, 1))  # only 6 months, historical has 12

    with pytest.raises(ValueError, match="fixed_control_points"):
        splice_fixed_columns(generated, ["free_a"], historical, config, site_names)


def test_overwrite_fixed_columns_replaces_only_fixed_columns():
    config = _basin_config(["fixed_a"])
    site_names = ["free_a", "fixed_a"]
    historical = _historical_df(12, site_names)

    synth_flow = pd.DataFrame(
        {"free_a": np.full(12, -1.0), "fixed_a": np.full(12, -1.0)},
        index=pd.date_range("2020-01", periods=12, freq="MS"),
    )
    result = overwrite_fixed_columns(synth_flow, historical, config)

    np.testing.assert_allclose(result["free_a"].to_numpy(), -1.0)
    np.testing.assert_allclose(result["fixed_a"].to_numpy(), historical["fixed_a"].to_numpy())


def test_overwrite_fixed_columns_raises_on_length_mismatch():
    config = _basin_config(["fixed_a"])
    site_names = ["free_a", "fixed_a"]
    historical = _historical_df(12, site_names)
    synth_flow = pd.DataFrame(
        {"free_a": np.zeros(6), "fixed_a": np.zeros(6)},
        index=pd.date_range("2020-01", periods=6, freq="MS"),
    )

    with pytest.raises(ValueError, match="fixed_control_points"):
        overwrite_fixed_columns(synth_flow, historical, config)


def test_overwrite_fixed_columns_no_fixed_sites_is_noop():
    config = _basin_config([])
    site_names = ["a", "b"]
    historical = _historical_df(6, site_names)
    synth_flow = pd.DataFrame(
        {"a": np.full(6, 3.0), "b": np.full(6, 4.0)},
        index=pd.date_range("2020-01", periods=6, freq="MS"),
    )

    result = overwrite_fixed_columns(synth_flow, historical, config)

    np.testing.assert_allclose(result["a"].to_numpy(), 3.0)
    np.testing.assert_allclose(result["b"].to_numpy(), 4.0)

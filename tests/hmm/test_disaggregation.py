"""
Tests for toolkit.hmm.disaggregation.disaggregate_annual_to_monthly.

These tests encode the *expected* behavior of the KNN spatio-temporal disaggregation
algorithm (anchor-conservation, ratio-preservation, nearest-neighbor selection,
reproducibility) using small hand-constructed historical datasets, rather than just
snapshotting whatever the current implementation happens to output.
"""
import numpy as np
import pytest

from toolkit.hmm.disaggregation import disaggregate_annual_to_monthly


def _uniform_monthly_history(annual_totals):
    """Build (hist_years*12,) monthly data where each year's total is spread evenly
    across its 12 months (annual_totals[i] / 12 per month)."""
    monthly = np.concatenate([np.full(12, total / 12) for total in annual_totals])
    return monthly


def _historical_data_with_site_ratios(annual_totals, site_ratios):
    """Build (hist_years*12, n_sites) historical data. Column 0 is the anchor, with
    annual_totals[i]/12 per month in year i. Each other column is a constant multiple
    (site_ratios[k]) of the anchor in every month of every year."""
    anchor = _uniform_monthly_history(annual_totals)
    columns = [anchor] + [anchor * ratio for ratio in site_ratios]
    return np.stack(columns, axis=1)


class _NearestChoiceRNG:
    """Stub RNG that deterministically selects the nearest candidate (first entry of
    the k-nearest-neighbor index array, which disaggregate_annual_to_monthly always
    passes in ascending-distance order). Used to test the neighbor-selection logic
    itself, independent of probabilistic sampling."""

    def choice(self, a, p=None):
        return a[0]


def test_deterministic_with_single_historical_year():
    """With one historical year, k = int(sqrt(1)) = 1, so there is only one possible
    neighbor - the algorithm is fully deterministic regardless of rng. Expected output
    is hand-computable directly from the single historical year's shape."""
    anchor_hist_monthly = np.arange(1, 13) * 10.0  # 10, 20, ..., 120 ; annual = 780
    historical = np.stack([anchor_hist_monthly, anchor_hist_monthly * 2], axis=1)

    synthetic_annual = np.array([390.0])  # exactly half of the historical annual total

    result = disaggregate_annual_to_monthly(
        annual_values=synthetic_annual,
        historical_monthly_data=historical,
        anchor_index=0,
        rng=np.random.default_rng(0),
    )

    expected_anchor = anchor_hist_monthly / 2  # scaled by 390/780 = 0.5
    expected_site1 = expected_anchor * 2  # constant 2x historical ratio is preserved

    np.testing.assert_allclose(result["site_0"].values, expected_anchor)
    np.testing.assert_allclose(result["site_1"].values, expected_site1)


def test_nearest_neighbor_is_selected():
    """With 4 historical years (k=2) and a stub rng that always picks the nearest
    candidate, the disaggregation should use the single historical year whose annual
    anchor total is closest to the synthetic value - not just any of the k nearest."""
    annual_totals = [100.0, 5000.0, 9000.0, 13000.0]  # year 1 (5000) is nearest to 5100
    historical = _historical_data_with_site_ratios(annual_totals, site_ratios=[3.0])

    result = disaggregate_annual_to_monthly(
        annual_values=np.array([5100.0]),
        historical_monthly_data=historical,
        anchor_index=0,
        rng=_NearestChoiceRNG(),
    )

    # Nearest historical year (5000) has a uniform monthly shape, so the synthetic
    # value should be split evenly across all 12 months: 5100 / 12 = 425.
    np.testing.assert_allclose(result["site_0"].values, np.full(12, 425.0))
    # Site 1 has a constant 3x ratio to the anchor in the historical data.
    np.testing.assert_allclose(result["site_1"].values, np.full(12, 1275.0))


@pytest.mark.parametrize("anchor_index", [0, 1, 2])
def test_anchor_conservation_for_any_anchor_position(anchor_index):
    """The anchor site's 12 disaggregated monthly values must always sum to the input
    annual value, for every year, regardless of which historical neighbor gets sampled
    and regardless of which column is used as the anchor (the point of this refactor
    is that the anchor no longer has to be the basin outlet / last column)."""
    rng = np.random.default_rng(123)
    annual_totals = [1000.0 * (i + 1) for i in range(9)]  # 9 years -> k = 3
    sites = [_uniform_monthly_history(annual_totals) for _ in range(3)]
    sites[anchor_index] = _uniform_monthly_history(annual_totals)
    historical = np.stack(sites, axis=1)

    synthetic_annual = np.array([1500.0, 4200.0, 8800.0])

    result = disaggregate_annual_to_monthly(
        annual_values=synthetic_annual,
        historical_monthly_data=historical,
        anchor_index=anchor_index,
        rng=rng,
    )

    anchor_col = result[f"site_{anchor_index}"].values.reshape(-1, 12)
    np.testing.assert_allclose(anchor_col.sum(axis=1), synthetic_annual)


@pytest.mark.parametrize("anchor_index", [0, 2])
def test_ratio_preservation_for_any_anchor_position(anchor_index):
    """If a site's historical monthly value is always a constant multiple of the
    anchor's, that exact ratio must be preserved in every disaggregated month/year -
    regardless of which historical neighbor is sampled."""
    rng = np.random.default_rng(7)
    annual_totals = [1000.0 * (i + 1) for i in range(9)]  # k = 3

    other_index = 1 if anchor_index != 1 else 0
    ratio = 2.5
    cols = {}
    anchor_hist = _uniform_monthly_history(annual_totals)
    cols[anchor_index] = anchor_hist
    cols[other_index] = anchor_hist * ratio
    # third column (unused in the assertion) just needs to exist
    remaining = [i for i in range(3) if i not in cols][0]
    cols[remaining] = anchor_hist
    historical = np.stack([cols[i] for i in range(3)], axis=1)

    synthetic_annual = np.array([1300.0, 6600.0])

    result = disaggregate_annual_to_monthly(
        annual_values=synthetic_annual,
        historical_monthly_data=historical,
        anchor_index=anchor_index,
        rng=rng,
    )

    np.testing.assert_allclose(
        result[f"site_{other_index}"].values, result[f"site_{anchor_index}"].values * ratio
    )


def test_output_shape_and_columns():
    annual_totals = [1000.0 * (i + 1) for i in range(9)]
    historical = _historical_data_with_site_ratios(annual_totals, site_ratios=[1.5, 0.5])
    synthetic_annual = np.array([2000.0, 3000.0, 4000.0])

    result = disaggregate_annual_to_monthly(
        annual_values=synthetic_annual,
        historical_monthly_data=historical,
        anchor_index=0,
        rng=np.random.default_rng(1),
    )

    assert result.shape == (len(synthetic_annual) * 12, historical.shape[1])
    assert list(result.columns) == ["site_0", "site_1", "site_2"]


def test_reproducible_with_same_seed():
    """The same seed must always produce the same output."""
    annual_totals = [1000.0 * (i + 1) for i in range(9)]  # k = 3
    historical = _historical_data_with_site_ratios(annual_totals, site_ratios=[1.7])
    synthetic_annual = np.linspace(1000, 9000, 15) + 250  # avoid exact ties

    result_a = disaggregate_annual_to_monthly(
        annual_values=synthetic_annual,
        historical_monthly_data=historical,
        anchor_index=0,
        rng=np.random.default_rng(42),
    )
    result_b = disaggregate_annual_to_monthly(
        annual_values=synthetic_annual,
        historical_monthly_data=historical,
        anchor_index=0,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(result_a.values, result_b.values)


def test_different_seeds_produce_different_output():
    """Different seeds should (almost certainly) sample different neighbors somewhere
    across many years, proving the rng is actually exercised rather than ignored."""
    annual_totals = [1000.0 * (i + 1) for i in range(9)]  # k = 3, 3 candidates per year
    historical = _historical_data_with_site_ratios(annual_totals, site_ratios=[1.7])
    synthetic_annual = np.linspace(1000, 9000, 15) + 250

    result_a = disaggregate_annual_to_monthly(
        annual_values=synthetic_annual,
        historical_monthly_data=historical,
        anchor_index=0,
        rng=np.random.default_rng(1),
    )
    result_b = disaggregate_annual_to_monthly(
        annual_values=synthetic_annual,
        historical_monthly_data=historical,
        anchor_index=0,
        rng=np.random.default_rng(2),
    )

    assert not np.array_equal(result_a.values, result_b.values)


def test_default_rng_used_when_none_provided():
    """Calling without an rng should not raise and should return valid output (a fresh
    default_rng() is used internally rather than mutating global numpy random state)."""
    annual_totals = [1000.0 * (i + 1) for i in range(9)]
    historical = _historical_data_with_site_ratios(annual_totals, site_ratios=[1.0])

    result = disaggregate_annual_to_monthly(
        annual_values=np.array([4000.0]),
        historical_monthly_data=historical,
        anchor_index=0,
    )
    assert result.shape == (12, 2)

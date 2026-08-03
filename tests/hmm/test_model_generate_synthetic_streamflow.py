"""
Tests that BayesianStreamflowHMM.generate_synthetic_streamflow's synthetic horizon is driven
by its explicit `num_years` parameter, independent of how many candidate blocks are in the
`historical_monthly_data` stencil pool.

`num_years` used to be silently derived from `historical_monthly_data.shape[0] // 12` -- fine
when the stencil pool was one historical record sized close to the desired synthetic horizon,
but wrong once the pool is a much larger multi-realization candidate set (see
toolkit.data.ninetyfiveofive.load_9505_stencil_pool).
"""
import arviz as az
import numpy as np
import pandas as pd
import pytest

from toolkit.hmm.model import BayesianStreamflowHMM


def _fitted_model(n_states=2):
    model = BayesianStreamflowHMM(n_states=n_states, random_seed=0)
    posterior = {
        "mu": np.array([[[0.5, 2.0]]]),
        "sigma": np.array([[[0.2, 0.2]]]),
        "transition_mat": np.array([[[[0.8, 0.2], [0.3, 0.7]]]]),
        "initial_dist": np.array([[[0.5, 0.5]]]),
    }
    model.idata = az.from_dict(posterior=posterior)
    return model


def _reference_single_pass_generate(
    model, start_year, num_years, historical_monthly_data, n_ensembles, random_seed, outflow_index
):
    """Oracle mirroring the pre-refactor implementation: annual generation and monthly
    disaggregation interleaved in a single loop per ensemble member, rather than the
    two-pass structure (all annual trajectories, then bias correction, then all
    disaggregation) added to support `bias_correction_method`. Used to confirm the
    two-pass version reproduces this exactly when bias correction is off."""
    np.random.seed(random_seed)
    disaggregation_rng = np.random.default_rng(random_seed)

    n_months = num_years * 12
    n_locations = historical_monthly_data.shape[1]
    streamflow = np.zeros((n_months, n_locations, n_ensembles))
    annual_states = np.zeros((num_years, n_ensembles), dtype=int)
    hmm_params = []

    for ens in range(n_ensembles):
        chains = model.idata.posterior.sizes['chain']
        draws = model.idata.posterior.sizes['draw']
        random_chain = np.random.choice(chains)
        random_draw = np.random.choice(draws)
        posterior_sample = model.idata.posterior.sel(chain=random_chain, draw=random_draw)
        mu = posterior_sample["mu"].values.squeeze()
        sigma = posterior_sample["sigma"].values.squeeze()
        transition_mat = posterior_sample["transition_mat"].values.squeeze()
        initial_dist = posterior_sample["initial_dist"].values.squeeze()
        param_vec = np.concatenate([
            mu.flatten(), sigma.flatten(),
            transition_mat[0, :].flatten(), transition_mat[1, :].flatten(),
            initial_dist.flatten(),
        ])
        hmm_params.append(param_vec)

        annual_synthetic = np.zeros(num_years)
        states = np.zeros(num_years, dtype=int)
        states[0] = np.random.choice(model.n_states, p=initial_dist)
        annual_synthetic[0] = np.random.normal(mu[states[0]], sigma[states[0]])
        for t in range(1, num_years):
            current_state = states[t - 1]
            probs = transition_mat[current_state]
            states[t] = np.random.choice(model.n_states, p=probs)
            annual_synthetic[t] = np.random.normal(mu[states[t]], sigma[states[t]])
        annual_synthetic = np.expm1(annual_synthetic)

        synth_monthly = model.disaggregate_annual_streamflow(
            annual_streamflow=annual_synthetic,
            historical_monthly_data=historical_monthly_data,
            outflow_index=outflow_index,
            rng=disaggregation_rng,
        )
        streamflow[:, :, ens] = synth_monthly.values if hasattr(synth_monthly, "values") else synth_monthly
        annual_states[:, ens] = states

    hmm_params = np.stack(hmm_params, axis=0)
    streamflow_out = np.transpose(streamflow, (2, 0, 1))
    return {
        "streamflow": streamflow_out,
        "annual_states": annual_states.T,
        "realization_meta": hmm_params,
    }


def test_synthetic_horizon_follows_num_years_not_stencil_pool_size():
    model = _fitted_model()

    n_sites = 2
    stencil_pool_years = 5  # candidate pool: larger than the desired synthetic horizon
    num_years = 3  # desired synthetic horizon
    n_ensembles = 2

    historical_monthly_data = np.random.default_rng(0).uniform(
        1, 100, size=(stencil_pool_years * 12, n_sites)
    )
    time_index = pd.date_range("2020-01", periods=num_years * 12, freq="MS")

    result = model.generate_synthetic_streamflow(
        start_year=2020,
        num_years=num_years,
        historical_monthly_data=historical_monthly_data,
        n_ensembles=n_ensembles,
        random_seed=42,
        site_names=[f"site_{i}" for i in range(n_sites)],
        time_index=list(time_index),
        outflow_index=0,
    )

    assert result["streamflow"].shape == (n_ensembles, num_years * 12, n_sites)
    assert result["annual_states"].shape == (n_ensembles, num_years)
    assert len(result["annual_states_index"]) == num_years


def test_bias_correction_off_matches_reference_single_pass_implementation():
    """Guards the loop-split added for bias correction: with bias_correction_method=None,
    the two-pass implementation (all annual trajectories, then disaggregation) must be
    bit-for-bit identical to the pre-refactor single interleaved loop, since disaggregation
    draws from its own independent rng that never interacts with the global numpy random
    state used for posterior/state sampling."""
    n_sites = 2
    num_years = 5
    n_ensembles = 4
    historical_monthly_data = np.random.default_rng(1).uniform(1, 100, size=(6 * 12, n_sites))
    time_index = pd.date_range("2020-01", periods=num_years * 12, freq="MS")
    kwargs = dict(
        start_year=2020,
        num_years=num_years,
        historical_monthly_data=historical_monthly_data,
        n_ensembles=n_ensembles,
        random_seed=42,
        outflow_index=0,
    )

    result = _fitted_model().generate_synthetic_streamflow(
        site_names=[f"site_{i}" for i in range(n_sites)],
        time_index=list(time_index),
        **kwargs,
    )
    reference = _reference_single_pass_generate(_fitted_model(), **kwargs)

    np.testing.assert_array_equal(result["streamflow"], reference["streamflow"])
    np.testing.assert_array_equal(result["annual_states"], reference["annual_states"])
    np.testing.assert_array_equal(result["realization_meta"], reference["realization_meta"])


def test_bias_correction_delta_scales_annual_totals_before_disaggregation():
    """Confirms bias correction actually reaches the disaggregation step: the corrected
    ensemble's outflow-site annual totals should match the historical mean used for delta
    scaling, which the fixture's raw (uncorrected) HMM parameters are nowhere near."""
    n_sites = 2
    num_years = 5
    n_ensembles = 3
    historical_monthly_data = np.random.default_rng(2).uniform(1, 100, size=(6 * 12, n_sites))
    time_index = pd.date_range("2020-01", periods=num_years * 12, freq="MS")
    historical_annual = np.array([500.0, 520.0, 480.0, 510.0, 495.0, 505.0])

    result = _fitted_model().generate_synthetic_streamflow(
        start_year=2020,
        num_years=num_years,
        historical_monthly_data=historical_monthly_data,
        n_ensembles=n_ensembles,
        random_seed=42,
        site_names=[f"site_{i}" for i in range(n_sites)],
        time_index=list(time_index),
        outflow_index=0,
        bias_correction_method="delta",
        historical_annual=historical_annual,
    )

    streamflow = result["streamflow"]  # (n_ensembles, n_months, n_sites)
    outflow_annual = streamflow[:, :, 0].reshape(n_ensembles, num_years, 12).sum(axis=2)

    assert outflow_annual.mean() == pytest.approx(historical_annual.mean(), rel=1e-6)


def test_bias_correction_requires_historical_annual():
    model = _fitted_model()
    with pytest.raises(ValueError, match="historical_annual"):
        model.generate_synthetic_streamflow(
            start_year=2020,
            num_years=3,
            historical_monthly_data=np.random.default_rng(0).uniform(1, 100, size=(5 * 12, 2)),
            n_ensembles=2,
            random_seed=42,
            site_names=["site_0", "site_1"],
            time_index=list(pd.date_range("2020-01", periods=3 * 12, freq="MS")),
            outflow_index=0,
            bias_correction_method="delta",
        )

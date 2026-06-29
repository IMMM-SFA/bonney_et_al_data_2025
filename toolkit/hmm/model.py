import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.scan.basic import scan
from typing import Optional, Dict, Any, Tuple
import logging
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import h5py

from toolkit.wrap.io import flo_to_df
from toolkit.graphics.hmm import plot_results, plot_diagnostics, plot_hmm_diagnostics
from toolkit.hmm.disaggregation import disaggregate_annual_to_monthly

logger = logging.getLogger(__name__)

class BayesianStreamflowHMM:
    """
    Bayesian Hidden Markov Model for streamflow modeling using forward algorithm.
    Implements complete pooling across ensemble members with efficient MCMC sampling.
    """
   
    def __init__(
        self,
        n_states: int = 2,
        random_seed: Optional[int] = None,
        prior_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Bayesian HMM model.
       
        Parameters
        ----------
        n_states : int, default=2
            Number of hidden states (dry/wet)
        random_seed : Optional[int], default=None
            Random seed for reproducibility
        prior_config : Optional[Dict[str, Any]], default=None
            Configuration for prior distributions
        """
        self.n_states = n_states
        self.random_seed = random_seed
        self.model = None
        self.idata = None
        self.prior_config = prior_config or {}
   
    def _build_model(self, data: np.ndarray) -> None:
        """
        Build the PyMC model using forward algorithm for efficient HMM likelihood computation.
        All ensemble members' data is treated as one dataset to train a single HMM.
       
        Parameters
        ----------
        data : np.ndarray
            Streamflow data with shape (n_members, n_years)
        """
        n_members, n_years = data.shape
       
        self.model = pm.Model()
        with self.model:
            # Emission mu
            prior_means = np.array(self.prior_config["mu_hyper"]["mu"])
            prior_sigma = self.prior_config["mu_hyper"]["sigma"]
            
            mu = pm.Normal(
                "mu",
                mu=prior_means,
                sigma=prior_sigma,
                shape=self.n_states
            )
           
            # Emission sigma
            sigma_hyper = self.prior_config["sigma_hyper"]["sigma"]
            sigma = pm.HalfNormal(
                "sigma",
                sigma=sigma_hyper,
                shape=self.n_states
            )
           
           # Transition matrix
            alpha = self.prior_config["transition"]["alpha"]
            beta = self.prior_config["transition"]["beta"]
            
            transition_priors = np.ones((self.n_states, self.n_states)) * beta
            np.fill_diagonal(transition_priors, alpha)
            
            transition_mat = pm.Dirichlet(
                "transition_mat",
                a=transition_priors,
                shape=(self.n_states, self.n_states)
            )
           
            # Initial state distribution
            initial_dist = pm.Dirichlet(
                "initial_dist",
                a=np.ones(self.n_states),
                shape=self.n_states
            )
           
            # Order constraint to prevent label switching
            pm.Potential("order_constraint", -pt.maximum(0, mu[0] - mu[1]) * 1e6)
           
            # Convert data to PyTensor constant for dimshuffle operations
            data_pt = pt.as_tensor_variable(data)
           
            # Vectorized forward algorithm for all ensemble members
            # Define forward step function that processes all members at once
            def hmm_step_vectorized(obs_t_all, prev_logp_all, trans_p, mu, sigma):
                """
                Vectorized forward step for all ensemble members simultaneously.
                
                Parameters
                ----------
                obs_t_all : vector (n_members,)
                    Current observations for all members
                prev_logp_all : matrix (n_members, n_states)
                    Previous log probabilities for all members and states
                trans_p : matrix (n_states, n_states)
                    Transition probability matrix
                mu : vector (n_states,)
                    State-specific emission means
                sigma : vector (n_states,)
                    State-specific emission standard deviations
                
                Returns
                -------
                matrix (n_members, n_states)
                    Current log probabilities for all members and states
                """
                # Emission log probabilities: (n_members, n_states)
                emission_logp = pm.Normal.logp(
                    obs_t_all.dimshuffle(0, 'x'),  # (n_members, 1)
                    mu.dimshuffle('x', 0),          # (1, n_states)
                    sigma.dimshuffle('x', 0)        # (1, n_states)
                )
                
                # Forward step for all members: (n_members, n_states)
                logp_t_all = pm.logsumexp(
                    prev_logp_all.dimshuffle(0, 1, 'x') +  # (n_members, n_states, 1)
                    pt.log(trans_p + 1e-10).dimshuffle('x', 0, 1) +  # (1, n_states, n_states)
                    emission_logp.dimshuffle(0, 'x', 1),    # (n_members, 1, n_states)
                    axis=1
                )
                return logp_t_all
            
            # Initial log probabilities for all members: (n_members, n_states)
            initial_logp_all = (
                pt.log(initial_dist).dimshuffle('x', 0) +  # (1, n_states)
                pm.Normal.logp(
                    data_pt[:, 0].dimshuffle(0, 'x'),  # (n_members, 1)
                    mu.dimshuffle('x', 0),           # (1, n_states)
                    sigma.dimshuffle('x', 0)         # (1, n_states)
                )
            )
            
            # Single scan for all members: sequences is (n_years-1, n_members)
            logp_all, _ = scan(
                fn=hmm_step_vectorized,
                sequences=data_pt[:, 1:].T,  # Transpose to (n_years-1, n_members)
                outputs_info=initial_logp_all,
                non_sequences=[transition_mat, mu, sigma]
            )
            
            # Total log-likelihood summed across all members: scalar
            total_logp_all = pt.sum(pm.logsumexp(logp_all[-1], axis=1))
            pm.Potential("likelihood_all", total_logp_all)
   
    def fit(
        self,
        data: np.ndarray,
        draws: int = 2000,
        tune: int = 2000,
        chains: int = 4,
        target_accept: float = 0.95,
        sampler: str = "nuts",
    ) -> None:
        """
        Fit the HMM model using MCMC.
       
        Parameters
        ----------
        data : np.ndarray
            Streamflow data with shape (n_members, n_years)
        draws : int, default=2000
            Number of posterior samples to draw
        tune : int, default=2000
            Number of tuning steps
        chains : int, default=4
            Number of MCMC chains
        target_accept : float, default=0.95
            Target acceptance rate for NUTS sampler
        sampler : str, default="nuts"
            Sampler to use: "nuts", "metropolis", "smc"
        """
        logger.info("Building model...")
        self._build_model(data)
       
        logger.info(f"Starting MCMC with {sampler} sampler...")
        with self.model:
            if sampler.lower() == "nuts":
                # Use NUTS sampler (gradient-based, most efficient)
                self.idata = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    random_seed=self.random_seed,
                    return_inferencedata=True,
                    target_accept=target_accept,
                    init="adapt_diag"
                )
            elif sampler.lower() == "smc":
                # Use Sequential Monte Carlo (most robust, no initialization issues)
                logger.info("Using SMC sampler - this may take longer but is very robust")
                self.idata = pm.sample_smc(
                    draws=draws,
                    chains=chains,
                    random_seed=self.random_seed,
                    return_inferencedata=True,
                )
            elif sampler.lower() == "metropolis":
                # Use Metropolis-Hastings (simple, slow but robust)
                logger.info("Using Metropolis sampler - slower but robust")
                self.idata = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    step=pm.Metropolis(),
                    random_seed=self.random_seed,
                    return_inferencedata=True,
                )
            else:
                raise ValueError(f"Unknown sampler: {sampler}. Choose 'nuts', 'smc', or 'metropolis'")
       
        # Check convergence
        rhat = az.rhat(self.idata)
        if not all(rhat < 1.01):
            logger.warning("Some parameters have R-hat > 1.01")
       
        # Calculate effective sample size
        ess = az.ess(self.idata)
        min_ess = float(ess.to_array().min().values)  # Convert to array first, then get min
        logger.info(f"Min ESS: {min_ess:.0f}")
   
    def predict_states(self, data: np.ndarray) -> np.ndarray:
        """
        Predict hidden states for given data.
       
        Parameters
        ----------
        data : np.ndarray
            Streamflow data with shape (n_members, n_years)
       
        Returns
        -------
        np.ndarray
            Predicted states with same shape as input data
        """
        logger.info("Predicting states...")
        if self.idata is None:
            raise ValueError("Model must be fit before making predictions")
       
        # Ensure data has correct shape
        if len(data.shape) == 1:
            data = data.reshape(1, -1)  # Add member dimension
        elif len(data.shape) == 3:
            # For 3D data, reshape to 2D by combining members and samples
            n_members, n_samples, n_years = data.shape
            data = data.reshape(n_members * n_samples, n_years)
       
        n_members, n_years = data.shape
        predicted_states = np.zeros((n_members, n_years), dtype=int)
       
        # Get posterior means of parameters
        mu = self.idata.posterior["mu"].mean(dim=("chain", "draw")).values
        sigma = self.idata.posterior["sigma"].mean(dim=("chain", "draw")).values
        transition_mat = self.idata.posterior["transition_mat"].mean(dim=("chain", "draw")).values
        initial_dist = self.idata.posterior["initial_dist"].mean(dim=("chain", "draw")).values
       
        # Viterbi algorithm for each member
        for m in range(n_members):
            predicted_states[m] = self._viterbi(
                data[m],
                mu,
                sigma,
                transition_mat,
                initial_dist
            )
       
        return predicted_states
   
    def _viterbi(self, observations: np.ndarray, means: np.ndarray,
                 sigmas: np.ndarray, transition_mat: np.ndarray,
                 initial_dist: np.ndarray) -> np.ndarray:
        """
        Viterbi algorithm for finding most likely state sequence.
       
        Parameters
        ----------
        observations : np.ndarray
            Observed streamflow values
        means : np.ndarray
            State-specific means
        sigmas : np.ndarray
            State-specific standard deviations
        transition_mat : np.ndarray
            Transition probability matrix
        initial_dist : np.ndarray
            Initial state distribution
           
        Returns
        -------
        np.ndarray
            Most likely state sequence
        """
        n_obs = len(observations)
        log_prob = np.zeros((n_obs, self.n_states))
        backpointer = np.zeros((n_obs, self.n_states), dtype=int)
       
        # Initial state probabilities using learned initial distribution
        log_prob[0] = np.log(initial_dist) + self._log_emission_prob(
            observations[0], means, sigmas
        )
       
        # Forward pass
        for t in range(1, n_obs):
            for j in range(self.n_states):
                emission_log_prob = self._log_emission_prob(
                    observations[t], means[j], sigmas[j]
                )
                trans_probs = log_prob[t-1] + np.log(transition_mat[:, j])
                best_prev = np.argmax(trans_probs)
                log_prob[t, j] = trans_probs[best_prev] + emission_log_prob
                backpointer[t, j] = best_prev
       
        # Backward pass
        states = np.zeros(n_obs, dtype=int)
        states[-1] = np.argmax(log_prob[-1])
        for t in range(n_obs-2, -1, -1):
            states[t] = backpointer[t+1, states[t+1]]
       
        return states
   
    def _log_emission_prob(self, x: float, means: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
        """
        Calculate log probability of emission for all states.
       
        Parameters
        ----------
        x : float
            Observed value
        means : np.ndarray
            State-specific means
        sigmas : np.ndarray
            State-specific standard deviations
           
        Returns
        -------
        np.ndarray
            Log probabilities for each state
        """
        return -0.5 * ((x - means) / sigmas)**2 - np.log(sigmas) - 0.5 * np.log(2 * np.pi)
   
    def save(self, filepath: str) -> None:
        """
        Save the model to disk.
       
        Parameters
        ----------
        filepath : str
            Path to save the model to
        """
        if self.idata is None:
            raise ValueError("Model must be fit before saving")
           
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
       
        # Save inference data
        self.idata.to_netcdf(f"{filepath}.nc")
       
        # Save model configuration
        config = {
            "n_states": self.n_states,
            "random_seed": self.random_seed,
            "prior_config": self.prior_config
        }
        with open(f"{filepath}_config.json", "w") as f:
            json.dump(config, f, indent=2)
   
    @classmethod
    def load(cls, filepath: str) -> "BayesianStreamflowHMM":
        """
        Load a saved model from disk.
       
        Parameters
        ----------
        filepath : str
            Path to load the model from
           
        Returns
        -------
        BayesianStreamflowHMM
            Loaded model instance
        """
        # Load configuration
        with open(f"{filepath}_config.json", "r") as f:
            config = json.load(f)
       
        # Create model instance
        model = cls(
            n_states=config["n_states"],
            random_seed=config["random_seed"],
            prior_config=config["prior_config"]
        )
       
        # Load inference data
        model.idata = az.from_netcdf(f"{filepath}.nc")
       
        return model

    def plot_results(self, data: np.ndarray, states: Optional[np.ndarray] = None, output_dir: Optional[Path] = None) -> None:
        """
        Plot model results and diagnostics.
       
        Parameters
        ----------
        data : np.ndarray
            Streamflow data with shape (n_members, n_years)
        states : Optional[np.ndarray], default=None
            Predicted states (if None, will use posterior mean states)
        output_dir : Optional[Path], default=None
            Directory to save plots. If None, saves to current directory.
        """
        plot_results(self.idata, data, states, self.n_states, output_dir)

    def plot_synthetic_comparison(
        self,
        historical_data: np.ndarray,
        ensemble_data: np.ndarray,
        n_years: int,
        n_samples: int = 5,
        output_dir: Optional[Path] = None
    ) -> None:
        """
        Plot comparison between historical, ensemble, and synthetic data.
       
        Parameters
        ----------
        historical_data : np.ndarray
            Historical streamflow data
        ensemble_data : np.ndarray
            Ensemble streamflow data
        n_years : int
            Number of years to plot
        n_samples : int, default=5
            Number of synthetic samples to generate
        output_dir : Optional[Path], default=None
            Directory to save plots. If None, plots are displayed.
        """
        logger.info("Generating synthetic comparison...")
        if self.idata is None:
            raise ValueError("Model must be fit before generating synthetic data")
           
        # Generate synthetic data
        synthetic = self.generate_annual_streamflow(
            n_years=n_years,
        )
       
        # Get states for synthetic data
        synthetic_states = np.zeros((n_samples, n_years), dtype=int)
        for i in range(n_samples):
            synthetic_states[i] = self.predict_states(synthetic[i:i+1])[0]
       
        plot_synthetic_comparison(
            self.idata,
            historical_data,
            ensemble_data,
            synthetic,
            synthetic_states,
            n_years,
            n_samples,
            output_dir=output_dir
        )

    def plot_diagnostics(self, output_dir: Optional[Path] = None) -> None:
        """
        Generate comprehensive diagnostic plots for the fitted model.
       
        Parameters
        ----------
        output_dir : Optional[Path], default=None
            Directory to save plots. If None, plots are displayed.
        """
        logger.info("Generating diagnostic plots...")
        plot_diagnostics(self.idata, output_dir)

    def plot_hmm_diagnostics(
        self, 
        data: np.ndarray, 
        output_dir: Optional[Path] = None
    ) -> None:
        """
        Generate HMM-specific diagnostic plots.
       
        Parameters
        ----------
        data : np.ndarray
            Data used for fitting the model
        output_dir : Optional[Path], default=None
            Directory to save plots. If None, plots are displayed.
        """
        logger.info("Generating HMM diagnostics...")
        plot_hmm_diagnostics(self.idata, data, output_dir)

    def generate_annual_streamflow(
        self,
        n_years: int,
        drought: Optional[float] = None,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate annual synthetic streamflow data using the HMM model.
       
        Parameters
        ----------
        n_years : int
            Number of years to generate
        drought : Optional[float], default=None
            Drought adjustment factor (0-1) to increase drought state likelihoods
        random_seed : Optional[int], default=None
            Random seed for reproducibility
           
        Returns
        -------
        np.ndarray
            Annual synthetic streamflow values
        """
        if self.idata is None:
            raise ValueError("Model must be fit before generating synthetic data")
           
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
           
        # Draw a single sample from the posterior distribution
        # Select both chain and draw randomly to get a single parameter set
        random_chain = np.random.choice(self.idata.posterior.sizes['chain'])
        random_draw = np.random.choice(self.idata.posterior.sizes['draw'])
        posterior_sample = self.idata.posterior.sel(chain=random_chain, draw=random_draw)
        
        # Extract parameters from the posterior sample and ensure correct dimensions
        mu = posterior_sample["mu"].values.squeeze()
        sigma = posterior_sample["sigma"].values.squeeze()
        transition_mat = posterior_sample["transition_mat"].values.squeeze()
        initial_dist = posterior_sample["initial_dist"].values.squeeze()
       
        # Adjust transition matrix for drought if specified
        if drought is not None:
            if mu[0] < mu[1]:  # identify which state is dry and which is wet
                dry_state = 0
                wet_state = 1
            else:
                dry_state = 1
                wet_state = 0
               
            # increase drought state likelihoods
            transition_mat[:, dry_state] = transition_mat[:, dry_state] + transition_mat[:, wet_state] * drought
            transition_mat[:, wet_state] = transition_mat[:, wet_state] - transition_mat[:, wet_state] * drought
           
            # Normalize transition matrix
            transition_mat = transition_mat / transition_mat.sum(axis=1, keepdims=True)
       
        # Generate annual synthetic data
        annual_synthetic = np.zeros(n_years)
        states = np.zeros(n_years, dtype=int)
       
        # Initial state using learned initial distribution
        states[0] = np.random.choice(self.n_states, p=initial_dist)
        annual_synthetic[0] = np.random.normal(mu[states[0]], sigma[states[0]])
       
        # Generate remaining years
        for t in range(1, n_years):
            # Get transition probabilities for current state
            current_state = states[t-1]
            probs = transition_mat[current_state]
            states[t] = np.random.choice(self.n_states, p=probs)
            annual_synthetic[t] = np.random.normal(mu[states[t]], sigma[states[t]])
           
        # Convert from log1p space back to original scale
        annual_synthetic = np.expm1(annual_synthetic)
           
        return annual_synthetic

    def generate_annual_streamflow_ensemble(
        self,
        n_years: int,
        n_samples: int = 100,
        drought: Optional[float] = None,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate an ensemble of annual synthetic streamflow data using the HMM model.
        This method draws multiple samples from the posterior to properly account for
        parameter uncertainty.
       
        Parameters
        ----------
        n_years : int
            Number of years to generate
        n_samples : int, default=100
            Number of posterior samples to draw
        drought : Optional[float], default=None
            Drought adjustment factor (0-1) to increase drought state likelihoods
        random_seed : Optional[int], default=None
            Random seed for reproducibility
           
        Returns
        -------
        np.ndarray
            Annual synthetic streamflow values with shape (n_samples, n_years)
        """
        if self.idata is None:
            raise ValueError("Model must be fit before generating synthetic data")
           
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
           
        # Initialize output array
        ensemble_synthetic = np.zeros((n_samples, n_years))
        
        # Generate samples
        for i in range(n_samples):
            ensemble_synthetic[i] = self.generate_annual_streamflow(
                n_years=n_years,
                drought=drought,
                random_seed=None  # Let each sample use different random seed
            )
           
        return ensemble_synthetic

    def disaggregate_annual_streamflow(
        self,
        annual_streamflow: np.ndarray,
        historical_monthly_data: np.ndarray,
        outflow_index: int,
        random_seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> pd.DataFrame:
        """
        Disaggregate annual streamflow data to monthly values using historical patterns.

        Thin wrapper around `toolkit.hmm.disaggregation.disaggregate_annual_to_monthly`.

        Parameters
        ----------
        annual_streamflow : np.ndarray
            Annual streamflow values to disaggregate
        historical_monthly_data : np.ndarray
            Historical monthly streamflow data with shape (n_years, 12, n_sites)
        outflow_index : int
            Index of the outflow control point site to use for distance calculation
        random_seed : Optional[int], default=None
            Random seed for reproducibility. Ignored if `rng` is provided.
        rng : Optional[np.random.Generator], default=None
            Random number generator for neighbor sampling. Takes precedence over `random_seed`.

        Returns
        -------
        pd.DataFrame
            Disaggregated monthly streamflow data
        """
        if rng is None:
            rng = np.random.default_rng(random_seed)

        return disaggregate_annual_to_monthly(
            annual_values=annual_streamflow,
            historical_monthly_data=historical_monthly_data,
            anchor_index=outflow_index,
            rng=rng,
        )

    def generate_synthetic_streamflow(
        self,
        start_year: int,
        # num_years: int,
        historical_monthly_data: np.ndarray,
        n_ensembles: int = 1000,
        drought: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
        h5_path: Optional[str] = None,
        site_names: Optional[list] = None,
        time_index: Optional[list] = None,
        outflow_index: int = -1
    ) -> None:
        """
        Generate multiple synthetic streamflow ensemble members and save to HDF5.
        Output axes: (ensemble_num, month_date, gage_site). Metadata for HMM parameters per member.
        
        Parameters
        ----------
        start_year : int
            Starting year for synthetic data
        historical_monthly_data : np.ndarray
            Historical monthly streamflow data
        n_ensembles : int, default=1000
            Number of ensemble members to generate
        drought : Optional[Dict[str, Any]], default=None
            Drought adjustment configuration
        random_seed : Optional[int], default=None
            Random seed for reproducibility
        h5_path : Optional[str], default=None
            Path to save HDF5 file
        site_names : Optional[list], default=None
            List of site names
        time_index : Optional[list], default=None
            Time index for streamflow
        outflow_index : int, default=-1
            Index of the outflow control point site for disaggregation
        
        Returns
        -------
        dict: Data dictionary containing:
            - 'streamflow': synthetic streamflow data (ensemble_num, month_date, gage_site)
            - 'annual_states': hidden states for each year (ensemble_num, year)
            - 'realization_meta': HMM parameters for each ensemble member
            - 'realization_meta_labels': parameter labels
            - 'streamflow_index': time index for streamflow
            - 'streamflow_columns': site names
            - 'annual_states_index': year index for states
        """
        from toolkit.data.io import dict_to_hdf5
        logger.info("Generating synthetic streamflow ensemble...")
        if self.idata is None:
            raise ValueError("Model must be fit before generating synthetic data")
        if random_seed is not None:
            np.random.seed(random_seed)
        disaggregation_rng = np.random.default_rng(random_seed)

        # reshape historical_monthly_data to (n_years, 12, n_locations)
        hist_monthly_sf = historical_monthly_data.reshape(historical_monthly_data.shape[0]//12, 12, -1)
        
        # Get number of years and months from historical_monthly_data
        num_years = hist_monthly_sf.shape[0]
        n_months = num_years * 12
        n_locations = hist_monthly_sf.shape[1] if len(hist_monthly_sf.shape) == 2 else hist_monthly_sf.shape[2]
        
        # Prepare output arrays
        streamflow = np.zeros((n_months, n_locations, n_ensembles))
        annual_states = np.zeros((num_years, n_ensembles), dtype=int)  # Store states for each year and ensemble
        hmm_params = []
        # Build detailed hmm_param_labels based on n_states
        hmm_param_labels = []
        for i in range(self.n_states):
            hmm_param_labels.append(f'mu_{i}')
        for i in range(self.n_states):
            hmm_param_labels.append(f'sigma_{i}')
        for i in range(self.n_states):
            for j in range(self.n_states):
                hmm_param_labels.append(f'transition_mat_{i}_{j}')
        for i in range(self.n_states):
            hmm_param_labels.append(f'initial_dist_{i}')
        # For each ensemble member, sample HMM parameters and generate synthetic data
        for ens in range(n_ensembles):
            # Randomly select a posterior sample (chain, draw)
            chains = self.idata.posterior.sizes['chain']
            draws = self.idata.posterior.sizes['draw']
            random_chain = np.random.choice(chains)
            random_draw = np.random.choice(draws)
            posterior_sample = self.idata.posterior.sel(chain=random_chain, draw=random_draw)
            mu = posterior_sample["mu"].values.squeeze()
            sigma = posterior_sample["sigma"].values.squeeze()
            transition_mat = posterior_sample["transition_mat"].values.squeeze()
            initial_dist = posterior_sample["initial_dist"].values.squeeze()
            # Store parameters for metadata
            param_vec = np.concatenate([
                mu.flatten(),
                sigma.flatten(),
                transition_mat[0,:].flatten(),
                transition_mat[1,:].flatten(),
                initial_dist.flatten()
            ])
            hmm_params.append(param_vec)
            # Generate annual synthetic data
            num_years = n_months // 12
            annual_synthetic = np.zeros(num_years)
            states = np.zeros(num_years, dtype=int)
            states[0] = np.random.choice(self.n_states, p=initial_dist)
            annual_synthetic[0] = np.random.normal(mu[states[0]], sigma[states[0]])
            for t in range(1, num_years):
                current_state = states[t-1]
                probs = transition_mat[current_state]
                states[t] = np.random.choice(self.n_states, p=probs)
                annual_synthetic[t] = np.random.normal(mu[states[t]], sigma[states[t]])
            annual_synthetic = np.expm1(annual_synthetic)
            # Disaggregate to monthly using a dedicated rng (advances across ensemble members,
            # independent of the global numpy random state used for posterior/state sampling above)
            synth_monthly = self.disaggregate_annual_streamflow(
                annual_streamflow=annual_synthetic,
                historical_monthly_data=historical_monthly_data,
                outflow_index=outflow_index,
                rng=disaggregation_rng,
            )
            # synth_monthly: (n_months, n_locations)
            streamflow[:, :, ens] = synth_monthly.values if hasattr(synth_monthly, 'values') else synth_monthly
            # Store the states for this ensemble member
            annual_states[:, ens] = states
        hmm_params = np.stack(hmm_params, axis=0)
        
        # Transpose streamflow to (ensemble_num, month_date, gage_site)
        streamflow_out = np.transpose(streamflow, (2, 0, 1))
        # Create year index for states
        year_index = [str(start_year + i) for i in range(num_years)]
        
        # Process time index
        time_index_str = [str(date.strftime('%Y-%m')) for date in time_index]
        
        # Save to HDF5 if requested
        data_dictionary = {
            'streamflow': streamflow_out,
            'annual_states': annual_states.T,  # Transpose to (ensemble, year) for consistency
            'realization_meta': hmm_params,
            'realization_meta_labels': hmm_param_labels,
            'streamflow_index': time_index_str,
            'streamflow_columns': site_names,
            'annual_states_index': year_index
        }
        return data_dictionary

from hmmlearn import hmm
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
        random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Disaggregate annual streamflow data to monthly values using historical patterns.
       
        Parameters
        ----------
        annual_streamflow : np.ndarray
            Annual streamflow values to disaggregate
        historical_monthly_data : np.ndarray
            Historical monthly streamflow data with shape (n_years, 12, n_sites)
        outflow_index : int
            Index of the outflow control point site to use for distance calculation
        random_seed : Optional[int], default=None
            Random seed for reproducibility
           
        Returns
        -------
        pd.DataFrame
            Disaggregated monthly streamflow data
        """
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
           
        num_years = len(annual_streamflow)
        hist_years = int(historical_monthly_data.shape[0] / 12)
        hist_monthly_sf = historical_monthly_data.reshape(hist_years, 12, -1)
        num_sites = hist_monthly_sf.shape[2]
       
        # Extract outflow control point data
        outflow_hist_monthly_sf = hist_monthly_sf[:, :, outflow_index]
        outflow_hist_annual_sf = np.sum(outflow_hist_monthly_sf, axis=1)
       
        # Compute distances between synthetic and historical annual flows using outflow node
        annual_distances = np.abs(
            np.subtract.outer(annual_streamflow, outflow_hist_annual_sf)
        )
       
        # Initialize synthetic monthly data
        synth_monthly_sf = np.zeros((num_years, 12, num_sites))
       
        # Compute spatial ratios: each site relative to outflow control point
        Vratios_mh = np.zeros(hist_monthly_sf.shape)
        for i in range(num_sites):
            Vratios_mh[:, :, i] = hist_monthly_sf[:, :, i] / outflow_hist_monthly_sf
       
        # Compute temporal breakdown: outflow monthly relative to outflow annual
        outflow_temporal_breakdown = np.zeros((hist_years, 12))
        for i in range(hist_years):
            outflow_temporal_breakdown[i, :] = (
                outflow_hist_monthly_sf[i, :] / outflow_hist_annual_sf[i]
            )
       
        # Compute neighbor probabilities (inverse distance weighting)
        k = int(np.sqrt(hist_years))
        neighbor_probabilities = np.zeros(k)
        for j in range(k):
            neighbor_probabilities[j] = 1 / (j + 1)
        neighbor_probabilities = neighbor_probabilities / np.sum(neighbor_probabilities)
       
        # Track zero values
        total_zeros = 0
       
        # For each synthetic year
        for j in range(num_years):
            # Find k nearest neighbors
            indices = np.argsort(annual_distances[j])[:k]
           
            # Select neighbor based on weighted probabilities
            neighbor_idx = np.random.choice(indices, p=neighbor_probabilities)
           
            # Step 1: Disaggregate outflow from annual to monthly (temporal)
            synth_monthly_sf[j, :, outflow_index] = (
                outflow_temporal_breakdown[neighbor_idx, :] * annual_streamflow[j]
            )
           
            # Step 2: Disaggregate across all sites (spatial)
            for n in range(12):
                synth_monthly_sf[j, n, :] = (
                    Vratios_mh[neighbor_idx, n, :] * synth_monthly_sf[j, n, outflow_index]
                )
                zeros = (Vratios_mh[neighbor_idx, n, :] == 0).sum()
                total_zeros += zeros
       
        # Reshape to (num_years * 12, num_sites)
        synth_monthly_sf = synth_monthly_sf.reshape(num_years * 12, num_sites)
       
        # Create DataFrame with datetime index
        start = "2000-01"  # Default start date, can be overridden by caller
        end = str(2000 + num_years - 1) + "-12"
        time_range = pd.date_range(start=start, end=end, freq="MS")
       
        # Create DataFrame
        monthly_streamflow = pd.DataFrame(
            synth_monthly_sf,
            index=time_range,
            columns=[f"site_{i}" for i in range(num_sites)]
        )
       
        return monthly_streamflow

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
            # Disaggregate to monthly (random_seed=None allows it to use the global random state)
            synth_monthly = self.disaggregate_annual_streamflow(
                annual_streamflow=annual_synthetic,
                historical_monthly_data=historical_monthly_data,
                outflow_index=outflow_index,
                random_seed=None
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


class WRAPStreamFlow:
    """Synthetic streamflow generation via HMM using WRAP .FLO file
    """
    def __init__(self, flo_file, outflow_name=None, ignore_columns=None):
        self.flo_file = flo_file
        self.outflow_name = outflow_name
        self.ignore_columns = ignore_columns

    def load_streamflows(self, start_year=None, end_year=None):
        self.monthly_flo_df = flo_to_df(self.flo_file)
        if self.ignore_columns is not None:
            self.monthly_flo_df = self.monthly_flo_df.loc[:, ~self.monthly_flo_df.columns.isin(self.ignore_columns)]
        if start_year is not None:
            self.monthly_flo_df = self.monthly_flo_df[self.monthly_flo_df.index.year >= start_year]
        if end_year is not None:
            self.monthly_flo_df = self.monthly_flo_df[self.monthly_flo_df.index.year >= end_year]
        self.annual_flo_df = self.monthly_flo_df.groupby(self.monthly_flo_df.index.year).sum()
        if self.outflow_name is not None:
            self.outflow_index = list(self.monthly_flo_df.columns).index(self.outflow_name)

    def _fit_HMM(self, random_seed, n_iter=1000):
        outflow_sf = self.annual_flo_df.loc[:, self.outflow_name].values
        # log_outflow_sf = np.log1p(outflow_sf)
        self.hmm = hmm.GaussianHMM(
            n_components=2, n_iter=n_iter, random_state=random_seed
        ).fit(outflow_sf.reshape(-1, 1))
        return self

    def _generate_synthetic_annual_sf(self, num_years, drought=None, random_state=None):
        if self.hmm is None:
            raise Exception("Model has not been fit.")

        # if drought parameter is provided, adjust transition parameters to increase drought.
        if drought is not None:
            if self.hmm.means_[0] < self.hmm.means_[1]: # identify which state is dry and which is wet
                dry_state = 0
                wet_state = 1
            else:
                dry_state = 1
                wet_state = 0

            # increase drought state likelihoods based on climate_drought_adjustment parameter
            self.hmm.transmat_[:,dry_state] = self.hmm.transmat_[:,dry_state] + self.hmm.transmat_[:,wet_state]* drought
            self.hmm.transmat_[:,wet_state] = self.hmm.transmat_[:,wet_state] - self.hmm.transmat_[:,wet_state]* drought

        log_annual_synthetic_outflow_sf = self.hmm.sample(num_years, random_state=random_state)
        # annual_synthetic_outflow_sf = np.exp(log_annual_synthetic_outflow_sf[0]) - 1
        annual_synthetic_outflow_sf = log_annual_synthetic_outflow_sf[0]

        return annual_synthetic_outflow_sf

    def _disaggregate_sf(self, annual_synthetic_outflow_sf, random_state=None):
        np.random.seed(random_state)
        outflow_hist_annual_sf = self.annual_flo_df.loc[:, self.outflow_name].values
        hist_years = outflow_hist_annual_sf.shape[0]
        synth_years = annual_synthetic_outflow_sf.shape[0]
        hist_monthly_sf = self.monthly_flo_df.values.reshape(hist_years, 12, -1)
        outflow_hist_monthly_sf = hist_monthly_sf[:, :, self.outflow_index]
        num_sites = len(self.monthly_flo_df.columns)

        # compute the similarities in outflow control point streamflow between synthetic and historical data
        annual_distances = abs(
            np.subtract.outer(annual_synthetic_outflow_sf, outflow_hist_annual_sf)
        )
        annual_distances = annual_distances.squeeze()

        # initialize the full synthetic data array
        synth_monthly_sf = np.zeros([synth_years, 12, num_sites])

        # compute ratios of flow between all control points and the outflow control points in historical data
        Vratios_mh = np.zeros(hist_monthly_sf.shape)
        for i in range(np.shape(hist_monthly_sf)[2]):
            Vratios_mh[:, :, i] = hist_monthly_sf[:, :, i] / outflow_hist_monthly_sf

        neighbor_probabilities = np.zeros([int(np.sqrt(hist_years))])
        # We sample from the square root of the number of years many neighbors
        for j in range(len(neighbor_probabilities)):
            neighbor_probabilities[j] = 1 / (j + 1)
        neighbor_probabilities = neighbor_probabilities / np.sum(neighbor_probabilities)

        outflow_temporal_breakdown = np.zeros([hist_years, 12])
        for i in range(outflow_hist_monthly_sf.shape[0]):
            outflow_temporal_breakdown[i, :] = (
                outflow_hist_monthly_sf[i, :] / outflow_hist_annual_sf[i]
            )
        total_zeros = 0
        for j in range(synth_years):
            # select one of k nearest neighbors for each simulated year
            indices = np.argsort(annual_distances[j, :])[
                0 : int(np.sqrt(hist_years))
            ]  # obtain nearest neighbor indices
            neighbor_index = np.random.choice(
                indices, 1, p=neighbor_probabilities
            )  # use probabilities to randomly choose a neighbor

            # use selected neighbor to disaggregate to monthly timescale
            synth_monthly_sf[j, :, -1] = (
                outflow_temporal_breakdown[neighbor_index, :]
                * annual_synthetic_outflow_sf[j]
            )

            # use selected neighbor to disagregate across gage sites
            for k in range(12):
                synth_monthly_sf[j, k, :] = (
                    Vratios_mh[neighbor_index, k, :] * synth_monthly_sf[j, k, -1]
                )
                zeros = (Vratios_mh[neighbor_index, k, :] == 0).sum()
                total_zeros += zeros

        synth_monthly_sf = np.reshape(synth_monthly_sf, (synth_years * 12, num_sites))
        return synth_monthly_sf

    def generate_synthetic_streamflow(self, start_year, num_years, drought=None, random_seed=None):
        self._fit_HMM(random_seed)
        annual_synthetic_outflow_sf = self._generate_synthetic_annual_sf(num_years, drought=drought, random_state=random_seed)
        synthetic_monthly_flow = self._disaggregate_sf(annual_synthetic_outflow_sf, random_state=random_seed)
        start = str(start_year) + "-01"
        end = str(int(start_year) + num_years-1) + "-12"
        time_range = pd.date_range(start=start, end=end, freq="MS")
        assert len(time_range) == synthetic_monthly_flow.shape[0]
        synthetic_streamflow = pd.DataFrame(
            synthetic_monthly_flow, columns=self.monthly_flo_df.columns, index=time_range
        )

        return synthetic_streamflow
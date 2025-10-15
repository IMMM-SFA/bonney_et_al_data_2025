from pathlib import Path
from typing import Dict, Any
import logging

from toolkit.wrap.io import flo_to_df
from toolkit.utils.random_seeds import get_seed
import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

def generate_prior_config_from_historical(
    hist_data: np.ndarray,
    n_states: int = 2,
    log1p_transform: bool = True
) -> Dict[str, Any]:
    """
    Generate prior configuration from historical data.
   
    Parameters
    ----------
    hist_data : np.ndarray
        Historical streamflow data
    n_states : int, default=2
        Number of hidden states
    log1p_transform : bool, default=True
        Whether data was log1p transformed
   
    Returns
    -------
    Dict[str, Any]
        Prior configuration dictionary
    """
    logger.info("Generating priors from historical data...")
    
    # Calculate basic statistics
    mean_flow = np.mean(hist_data)
    std_flow = np.std(hist_data)
    
    logger.info(f"Historical mean: {mean_flow:.3f}")
    logger.info(f"Historical std: {std_flow:.3f}")
    
    # Use K-means to get initial estimates of state means and standard deviations
    kmeans = KMeans(n_clusters=n_states, random_state=get_seed("kmeans_clustering"))
    state_labels = kmeans.fit_predict(hist_data.reshape(-1, 1))
    
    # Calculate state-specific statistics
    state_means = []
    state_stds = []
    for i in range(n_states):
        state_data = hist_data[state_labels == i]
        state_means.append(np.mean(state_data))
        state_stds.append(np.std(state_data))
    
    # Sort states by mean to ensure consistent ordering
    state_order = np.argsort(state_means)
    state_means = np.array(state_means)[state_order]
    state_stds = np.array(state_stds)[state_order]
    
    # Calculate transition probabilities from historical data
    transitions = np.zeros((n_states, n_states))
    for i in range(len(state_labels) - 1):
        transitions[state_labels[i], state_labels[i + 1]] += 1
    
    # Normalize transition counts to get probabilities
    row_sums = transitions.sum(axis=1, keepdims=True)
    transition_probs = np.divide(transitions, row_sums, where=row_sums != 0)
    
    # Calculate concentration parameters for Dirichlet prior
    # Use the observed transition probabilities to inform the prior
    # alpha: controls strength of self-transitions (higher = more persistent states)
    # beta: controls strength of cross-transitions (higher = more switching between states)
    alpha = np.max(transition_probs.diagonal()) * 10  # Strong self-transition prior
    beta = (1 - np.max(transition_probs.diagonal())) * 5  # Weaker cross-transition prior
    
    # Log the computed parameters
    logger.info(f"State means: {state_means}")
    logger.info(f"State stds: {state_stds}")
    logger.info(f"Alpha: {alpha:.3f}")
    logger.info(f"Beta: {beta:.3f}")
    
    # Create prior configuration with more flexible priors
    data_range = hist_data.max() - hist_data.min()
    data_std = hist_data.std()
    
    prior_config = {
        "mu_hyper": {
            "mu": state_means.tolist(),
            "sigma": max(data_std * 1.5, data_range * 0.2)
        },
        "sigma_hyper": {
            "sigma": max(data_std * 0.8, data_range * 0.1)
        },
        "transition": {
            "alpha": float(alpha),
            "beta": float(beta)
        }
    }
    
    return prior_config

def generate_prior_config_from_historical_legacy(
    flo_file: Path,
    gage_name: str,
    n_states: int = 2,
    log1p_transform: bool = False
) -> Dict[str, Any]:
    """
    Legacy function: Generate prior configuration for the Bayesian HMM based on historical data from FLO file.
    
    Parameters
    ----------
    flo_file : Path
        Path to C3.FLO file containing historical streamflow data
    gage_name : str
        Name of the gage (e.g., "INK20000")
    n_states : int, default=2
        Number of hidden states in the HMM
    log1p_transform : bool, default=False
        Whether the data has been log1p transformed
        
    Returns
    -------
    Dict[str, Any]
        Prior configuration dictionary
    """
    
    # Load historical data
    df = flo_to_df(flo_file)
    if gage_name not in df.columns:
        raise ValueError(f"Gage {gage_name} not found in C3.FLO file")
    
    # Get annual streamflow
    annual_flow = df[gage_name].resample('Y').sum()
    flow_data = annual_flow.values
    
    # Apply log1p transformation if requested
    if log1p_transform:
        flow_data = np.log1p(flow_data)
    
    # Use K-means to get initial estimates of state means and standard deviations
    kmeans = KMeans(n_clusters=n_states, random_state=get_seed("kmeans_clustering"))
    state_labels = kmeans.fit_predict(flow_data.reshape(-1, 1))
    
    # Calculate state-specific statistics
    state_means = []
    state_stds = []
    for i in range(n_states):
        state_data = flow_data[state_labels == i]
        state_means.append(np.mean(state_data))
        state_stds.append(np.std(state_data))
    
    # Sort states by mean to ensure consistent ordering
    state_order = np.argsort(state_means)
    state_means = np.array(state_means)[state_order]
    state_stds = np.array(state_stds)[state_order]
    
    # Calculate transition probabilities from historical data
    transitions = np.zeros((n_states, n_states))
    for i in range(len(state_labels) - 1):
        transitions[state_labels[i], state_labels[i + 1]] += 1
    
    # Normalize transition counts to get probabilities
    row_sums = transitions.sum(axis=1, keepdims=True)
    transition_probs = np.divide(transitions, row_sums, where=row_sums != 0)
    
    # Calculate concentration parameters for Dirichlet prior
    # Use the observed transition probabilities to inform the prior
    # alpha: controls strength of self-transitions (higher = more persistent states)
    # beta: controls strength of cross-transitions (higher = more switching between states)
    alpha = np.max(transition_probs.diagonal()) * 10  # Strong self-transition prior
    beta = (1 - np.max(transition_probs.diagonal())) * 5  # Weaker cross-transition prior
    
    # Create prior configuration
    prior_config = {
        "mu_hyper": {
            "mu": state_means.tolist(),
            "sigma": np.mean(state_stds)  # Use average state std as prior std
        },
        "sigma_hyper": {
            "sigma": np.mean(state_stds)  # Use average state std as prior std
        },
        "transition": {
            "alpha": float(alpha),
            "beta": float(beta)
        }
    }
    
    return prior_config 
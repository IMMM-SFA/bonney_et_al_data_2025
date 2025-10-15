"""
Utility functions for managing random seeds across the project.
"""
import json
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def load_random_seeds(config_path: Optional[Path] = None) -> Dict[str, int]:
    """
    Load random seed configuration from JSON file.
    
    Parameters
    ----------
    config_path : Optional[Path], default=None
        Path to random seeds configuration file. If None, uses default location.
        
    Returns
    -------
    Dict[str, int]
        Dictionary mapping seed names to seed values
    """
    if config_path is None:
        # Default path relative to this file
        config_path = Path(__file__).parent.parent.parent / "data" / "configs" / "random_seeds.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config["seeds"]
    except Exception as e:
        logger.warning(f"Could not load random seeds from {config_path}: {e}")
        # Return default seeds
        return {
            "hmm_training": 42,
            "hmm_generation": 42,
            "kmeans_clustering": 42,
            "plotting": 42,
            "multiprocessing": 42,
            "numpy_default": 42,
            "pytensor_default": 42,
            "pymc_default": 42
        }

def set_random_seeds(seed_name: str, config_path: Optional[Path] = None) -> int:
    """
    Set random seeds for all relevant libraries based on the specified seed name.
    
    Parameters
    ----------
    seed_name : str
        Name of the seed to use (e.g., 'hmm_training', 'plotting')
    config_path : Optional[Path], default=None
        Path to random seeds configuration file
        
    Returns
    -------
    int
        The seed value that was set
    """
    seeds = load_random_seeds(config_path)
    
    if seed_name not in seeds:
        logger.warning(f"Seed '{seed_name}' not found in configuration. Using default seed 42.")
        seed_value = 42
    else:
        seed_value = seeds[seed_name]
    
    # Set seeds for all relevant libraries
    np.random.seed(seed_value)
    random.seed(seed_value)
    
    # Try to set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
    except ImportError:
        pass
    
    # Try to set TensorFlow seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed_value)
    except ImportError:
        pass
    
    logger.info(f"Set random seeds to {seed_value} for '{seed_name}'")
    return seed_value

def get_seed(seed_name: str, config_path: Optional[Path] = None) -> int:
    """
    Get a specific seed value without setting it.
    
    Parameters
    ----------
    seed_name : str
        Name of the seed to retrieve
    config_path : Optional[Path], default=None
        Path to random seeds configuration file
        
    Returns
    -------
    int
        The seed value
    """
    seeds = load_random_seeds(config_path)
    return seeds.get(seed_name, 42)

def create_numpy_rng(seed_name: str, config_path: Optional[Path] = None) -> np.random.Generator:
    """
    Create a numpy random number generator with the specified seed.
    
    Parameters
    ----------
    seed_name : str
        Name of the seed to use
    config_path : Optional[Path], default=None
        Path to random seeds configuration file
        
    Returns
    -------
    np.random.Generator
        Numpy random number generator
    """
    seed_value = get_seed(seed_name, config_path)
    return np.random.default_rng(seed_value)

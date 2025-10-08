import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from typing import Optional
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def plot_results(idata, data: np.ndarray, states: Optional[np.ndarray] = None, n_states: int = 2, output_dir: Optional[Path] = None) -> None:
    """
    Plot model results and diagnostics.
    
    Parameters
    ----------
    idata : arviz.InferenceData
        Inference data from model fitting
    data : np.ndarray
        Streamflow data with shape (n_members, n_years)
    states : Optional[np.ndarray], default=None
        Predicted states (if None, will use posterior mean states)
    n_states : int, default=2
        Number of states in the model
    output_dir : Optional[Path], default=None
        Directory to save plots. If None, saves to current directory.
    """
    if idata is None:
        raise ValueError("Model must be fit before plotting results")
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot posterior traces
    az.plot_trace(
        idata,
        var_names=["mu", "sigma", "transition_mat"]
    )
    plt.suptitle("Parameter Posterior Traces\n" + 
                "For mu and sigma: Colors represent states (Dry/Wet), lines represent chains\n" +
                "For transition_mat: Colors represent transition probabilities (dry→dry, dry→wet, wet→dry, wet→wet)", 
                fontsize=18)
    plt.tight_layout()
    plt.savefig(output_dir / 'hmm_parameter_traces.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot state-specific parameters
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot state means
    mu = idata.posterior["mu"].mean(dim=("chain", "draw")).values
    mu_std = idata.posterior["mu"].std(dim=("chain", "draw")).values
    ax1.bar(range(n_states), mu, yerr=mu_std, capsize=5)
    ax1.set_title("State-specific Mean Streamflow", fontsize=16)
    ax1.set_xlabel("State", fontsize=14)
    ax1.set_ylabel("Mean Flow (acre-feet)", fontsize=14)
    ax1.set_xticks(range(n_states))
    ax1.set_xticklabels(["Dry", "Wet"])
    ax1.grid(True, alpha=0.3)
    
    # Plot state standard deviations
    sigma = idata.posterior["sigma"].mean(dim=("chain", "draw")).values
    sigma_std = idata.posterior["sigma"].std(dim=("chain", "draw")).values
    ax2.bar(range(n_states), sigma, yerr=sigma_std, capsize=5)
    ax2.set_title("State-specific Standard Deviation", fontsize=16)
    ax2.set_xlabel("State", fontsize=14)
    ax2.set_ylabel("Standard Deviation", fontsize=14)
    ax2.set_xticks(range(n_states))
    ax2.set_xticklabels(["Dry", "Wet"])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hmm_state_parameters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot posterior distributions
    az.plot_posterior(
        idata,
        var_names=["mu", "sigma", "transition_mat"]
    )
    plt.suptitle("Parameter Posterior Distributions", fontsize=18)
    plt.tight_layout()
    plt.savefig(output_dir / 'hmm_posterior_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot streamflow and states for first member
    if states is None:
        raise ValueError("States must be provided for plotting")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot streamflow
    ax1.plot(data[0], label="Observed Flow", alpha=0.7)
    ax1.set_title("Streamflow", fontsize=16)
    ax1.set_ylabel("Flow (acre-feet)", fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot states
    ax2.plot(states[0], label="Inferred States", alpha=0.7)
    ax2.set_title("Inferred States", fontsize=16)
    ax2.set_xlabel("Time", fontsize=14)
    ax2.set_ylabel("State", fontsize=14)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Dry", "Wet"])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hmm_streamflow_states.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_diagnostics(idata: az.InferenceData, output_dir: Optional[Path] = None) -> None:
    """
    Generate comprehensive diagnostic plots for the fitted model.
   
    Parameters
    ----------
    idata : az.InferenceData
        Inference data from PyMC
    output_dir : Optional[Path], default=None
        Directory to save plots. If None, plots are displayed.
    """
    logger.info("Generating diagnostic plots...")
    if idata is None:
        raise ValueError("Model must be fit before generating diagnostics")
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
    # 1. ENERGY PLOT - Sampling efficiency
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_energy(idata, ax=ax)
    ax.set_title("Energy Plot\n" + 
                "Blue: Energy distribution of samples\n" +
                "Orange: Energy transition distribution\n" +
                "Good overlap indicates efficient sampling", 
                fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'hmm_energy_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. TRACE PLOTS - Convergence visualization
    az.plot_trace(idata, var_names=["mu", "sigma", "transition_mat", "initial_dist"])
    plt.suptitle("Trace Plots - Parameter Convergence\n" +
                "Good convergence: chains mix well, no trends, stable", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'hmm_trace_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. R-HAT PLOTS - Convergence diagnostics
    fig, ax = plt.subplots(figsize=(10, 6))
    rhat = az.rhat(idata)
    rhat_array = rhat.to_array()
    var_names = rhat_array.coords['variable'].values
    rhat_values = []
    for var in var_names:
        var_rhat = rhat_array.sel(variable=var).values
        # Handle multi-dimensional arrays by taking the mean
        if var_rhat.size > 1:
            var_rhat = np.mean(var_rhat)
        else:
            var_rhat = float(var_rhat)
        rhat_values.append(var_rhat)
    
    bars = ax.bar(range(len(var_names)), rhat_values, color=['red' if x > 1.01 else 'green' for x in rhat_values])
    ax.axhline(y=1.01, color='red', linestyle='--', alpha=0.7, label='Convergence threshold')
    ax.set_title("R-hat Convergence Diagnostics\n" +
                "Green: Converged (< 1.01), Red: Potential issues (> 1.01)", fontsize=14)
    ax.set_xlabel("Parameters")
    ax.set_ylabel("R-hat value")
    ax.set_xticks(range(len(var_names)))
    ax.set_xticklabels(var_names, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'hmm_rhat_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. EFFECTIVE SAMPLE SIZE PLOTS
    fig, ax = plt.subplots(figsize=(10, 6))
    ess = az.ess(idata)
    ess_array = ess.to_array()
    ess_values = []
    for var in var_names:
        var_ess = ess_array.sel(variable=var).values
        # Handle multi-dimensional arrays by taking the mean
        if var_ess.size > 1:
            var_ess = np.mean(var_ess)
        else:
            var_ess = float(var_ess)
        ess_values.append(var_ess)
    
    bars = ax.bar(range(len(var_names)), ess_values, color=['red' if x < 100 else 'green' for x in ess_values])
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Minimum ESS threshold')
    ax.set_title("Effective Sample Size Diagnostics\n" +
                "Green: Sufficient (> 100), Red: Insufficient (< 100)", fontsize=14)
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Effective Sample Size")
    ax.set_xticks(range(len(var_names)))
    ax.set_xticklabels(var_names, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'hmm_ess_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. RANK PLOTS - Chain mixing
    az.plot_rank(idata, var_names=["mu", "sigma", "transition_mat", "initial_dist"])
    plt.suptitle("Rank Plots - Chain Mixing\n" +
                "Good mixing: uniform distribution across ranks", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'hmm_rank_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. AUTOCORRELATION PLOTS - Independence
    az.plot_autocorr(idata, var_names=["mu", "sigma", "transition_mat", "initial_dist"])
    plt.suptitle("Autocorrelation Plots - Sample Independence\n" +
                "Good: rapid decay to zero", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'hmm_autocorrelation_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. DIVERGENCE ANALYSIS
    divergence_rate = 0.0  # Initialize
    if hasattr(idata, 'sample_stats') and 'diverging' in idata.sample_stats:
        n_divergences = int(idata.sample_stats.diverging.sum().values)
        total_samples = int(idata.sample_stats.diverging.count().values)
        divergence_rate = n_divergences / total_samples if total_samples > 0 else 0
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(['Divergences', 'Good Samples'], 
               [n_divergences, total_samples - n_divergences],
               color=['red', 'green'])
        ax.set_title(f"Divergence Analysis\n" +
                    f"Rate: {divergence_rate:.2%} ({n_divergences}/{total_samples})", fontsize=14)
        ax.set_ylabel("Number of samples")
        plt.tight_layout()
        plt.savefig(output_dir / 'hmm_divergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 8. POSTERIOR DISTRIBUTIONS
    az.plot_posterior(idata, var_names=["mu", "sigma", "transition_mat", "initial_dist"])
    plt.suptitle("Posterior Distributions\n" +
                "Check for reasonable parameter ranges and shapes", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'hmm_posterior_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. PAIR PLOTS - Parameter correlations
    az.plot_pair(idata, var_names=["mu", "sigma"], marginals=True)
    plt.suptitle("Parameter Pair Plots\n" +
                "Check for parameter correlations and identifiability", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'hmm_pair_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 10. FOREST PLOTS - Parameter comparison
    az.plot_forest(idata, var_names=["mu", "sigma"], combined=True)
    plt.title("Forest Plot - Parameter Estimates\n" +
             "Shows posterior means and credible intervals", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'hmm_forest_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print comprehensive summary statistics
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL DIAGNOSTICS")
    print("="*60)
    
    print("\n1. CONVERGENCE SUMMARY:")
    if rhat_values:
        print(f"   Maximum R-hat: {max(rhat_values):.3f}")
    if ess_values:
        print(f"   Minimum ESS: {min(ess_values):.0f}")
    print(f"   Divergence rate: {divergence_rate:.2%}")
    
    print("\n2. PARAMETER SUMMARY:")
    print(az.summary(idata, var_names=["mu", "sigma", "transition_mat", "initial_dist"]))
    
    print("\n3. CONVERGENCE ASSESSMENT:")
    if rhat_values and ess_values:
        if max(rhat_values) < 1.01 and min(ess_values) > 100 and divergence_rate < 0.01:
            print("   ✅ EXCELLENT: All convergence criteria met")
        elif max(rhat_values) < 1.05 and min(ess_values) > 50 and divergence_rate < 0.05:
            print("   ⚠️  ACCEPTABLE: Some convergence issues, but usable")
        else:
            print("   ❌ POOR: Significant convergence issues detected")
    else:
        print("   ⚠️  Unable to assess convergence (missing data)")
    
    print("\n4. RECOMMENDATIONS:")
    if rhat_values and max(rhat_values) > 1.01:
        print("   - Increase number of draws and tune steps")
        print("   - Check for parameter identifiability issues")
    if ess_values and min(ess_values) < 100:
        print("   - Increase number of draws")
        print("   - Consider reparameterization")
    if divergence_rate > 0.01:
        print("   - Increase target_accept")
        print("   - Reparameterize the model")
        print("   - Check data preprocessing")
    
    print("="*60)

def plot_hmm_diagnostics(
    idata: az.InferenceData, 
    data: np.ndarray, 
    output_dir: Optional[Path] = None
) -> None:
    """
    Generate HMM-specific diagnostic plots.
   
    Parameters
    ----------
    idata : az.InferenceData
        Inference data from PyMC
    data : np.ndarray
        Data used for fitting the model
    output_dir : Optional[Path], default=None
        Directory to save plots. If None, plots are displayed.
    """
    logger.info("Generating HMM diagnostics...")
    if idata is None:
        raise ValueError("Model must be fit before generating diagnostics")
    
    try:
        # Set output directory
        if output_dir is None:
            output_dir = Path(".")
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get posterior means
        mu_mean = idata.posterior["mu"].mean(dim=("chain", "draw")).values
        sigma_mean = idata.posterior["sigma"].mean(dim=("chain", "draw")).values
        transition_mean = idata.posterior["transition_mat"].mean(dim=("chain", "draw")).values
        initial_mean = idata.posterior["initial_dist"].mean(dim=("chain", "draw")).values
    
        # 1. STATE TRANSITION ANALYSIS
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Transition matrix heatmap
        im = ax1.imshow(transition_mean, cmap='Blues', vmin=0, vmax=1)
        ax1.set_title("Transition Matrix (Posterior Mean)", fontsize=14)
        ax1.set_xlabel("To State")
        ax1.set_ylabel("From State")
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(["Dry", "Wet"])
        ax1.set_yticklabels(["Dry", "Wet"])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax1.text(j, i, f'{transition_mean[i, j]:.3f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax1)
        
        # Self-transition probabilities
        self_transitions = np.diag(transition_mean)
        ax2.bar(['Dry', 'Wet'], self_transitions, color=['orange', 'blue'])
        ax2.set_title("Self-Transition Probabilities", fontsize=14)
        ax2.set_ylabel("Probability")
        ax2.set_ylim(0, 1)
        for i, v in enumerate(self_transitions):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Initial state distribution
        ax3.bar(['Dry', 'Wet'], initial_mean, color=['orange', 'blue'])
        ax3.set_title("Initial State Distribution", fontsize=14)
        ax3.set_ylabel("Probability")
        ax3.set_ylim(0, 1)
        for i, v in enumerate(initial_mean):
            ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # State separation analysis
        mu_diff = abs(mu_mean[1] - mu_mean[0])
        mu_avg = (mu_mean[0] + mu_mean[1]) / 2
        separation = mu_diff / mu_avg if mu_avg > 0 else 0
        
        ax4.bar(['State Separation'], [separation], 
                color=['green' if separation > 0.1 else 'red'])
        ax4.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Good separation threshold')
        ax4.set_title(f"State Separation: {separation:.3f}", fontsize=14)
        ax4.set_ylabel("Separation ratio")
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'hmm_state_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
        # 2. EMISSION DISTRIBUTION DIAGNOSTICS
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # State means comparison
        ax1.bar(['Dry', 'Wet'], mu_mean, yerr=sigma_mean, capsize=5, 
                color=['orange', 'blue'], alpha=0.7)
        ax1.set_title("State-Specific Means", fontsize=14)
        ax1.set_ylabel("Mean Value")
        ax1.grid(True, alpha=0.3)
        
        # State standard deviations
        ax2.bar(['Dry', 'Wet'], sigma_mean, color=['orange', 'blue'], alpha=0.7)
        ax2.set_title("State-Specific Standard Deviations", fontsize=14)
        ax2.set_ylabel("Standard Deviation")
        ax2.grid(True, alpha=0.3)
        
        # Data distribution by state (simulated)
        # Generate samples from posterior to show state distributions
        n_samples = 1000
        state_samples = np.random.choice([0, 1], size=n_samples, p=initial_mean)
        emission_samples = np.random.normal(mu_mean[state_samples], sigma_mean[state_samples])
        
        ax3.hist(emission_samples[state_samples == 0], bins=30, alpha=0.7, 
                 label='Dry State', color='orange', density=True)
        ax3.hist(emission_samples[state_samples == 1], bins=30, alpha=0.7, 
                 label='Wet State', color='blue', density=True)
        ax3.set_title("Simulated Emission Distributions", fontsize=14)
        ax3.set_xlabel("Value")
        ax3.set_ylabel("Density")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Parameter uncertainty
        mu_std = idata.posterior["mu"].std(dim=("chain", "draw")).values
        sigma_std = idata.posterior["sigma"].std(dim=("chain", "draw")).values
        
        x = np.arange(2)
        width = 0.35
        
        ax4.bar(x - width/2, mu_std, width, label='μ uncertainty', alpha=0.7)
        ax4.bar(x + width/2, sigma_std, width, label='σ uncertainty', alpha=0.7)
        ax4.set_title("Parameter Uncertainty", fontsize=14)
        ax4.set_xlabel("State")
        ax4.set_ylabel("Standard Deviation")
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Dry', 'Wet'])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'hmm_emission_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
        # 3. MODEL FIT ASSESSMENT
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Data vs model comparison
        # Plot histogram of actual data
        data_flat = data.flatten()
        ax1.hist(data_flat, bins=30, alpha=0.7, label='Actual Data', density=True, color='gray')
        
        # Overlay model predictions - generate samples for the actual data length
        n_data_samples = len(data_flat)
        # Generate state samples for the data length
        data_state_samples = np.random.choice([0, 1], size=n_data_samples, p=initial_mean)
        model_samples = np.random.normal(mu_mean[data_state_samples], sigma_mean[data_state_samples])
        ax1.hist(model_samples, bins=30, alpha=0.7, label='Model Predictions', density=True, color='red')
        ax1.set_title("Data vs Model Distribution", fontsize=14)
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Density")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # QQ plot for normality check
        from scipy import stats
        stats.probplot(data_flat, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot (Normality Check)", fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Time series of data with state overlay (first ensemble member)
        ax3.plot(data[0], 'b-', alpha=0.7, label='Data')
        ax3.set_title("Time Series (First Ensemble Member)", fontsize=14)
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Value")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # State persistence analysis
        # Calculate expected state durations
        expected_durations = 1 / (1 - self_transitions)
        ax4.bar(['Dry', 'Wet'], expected_durations, color=['orange', 'blue'], alpha=0.7)
        ax4.set_title("Expected State Durations", fontsize=14)
        ax4.set_ylabel("Time Steps")
        ax4.grid(True, alpha=0.3)
        for i, v in enumerate(expected_durations):
            ax4.text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'hmm_model_fit.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print HMM-specific diagnostics
        print("\n" + "="*60)
        print("HMM-SPECIFIC DIAGNOSTICS")
        print("="*60)
        
        print(f"\n1. STATE ANALYSIS:")
        print(f"   State 0 (Dry) mean: {mu_mean[0]:.3f} ± {mu_std[0]:.3f}")
        print(f"   State 1 (Wet) mean: {mu_mean[1]:.3f} ± {mu_std[1]:.3f}")
        print(f"   State separation: {separation:.3f} {'✅' if separation > 0.1 else '⚠️'}")
        
        print(f"\n2. TRANSITION ANALYSIS:")
        print(f"   Self-transition probabilities: {self_transitions}")
        print(f"   Expected state durations: {expected_durations}")
        print(f"   Initial state distribution: {initial_mean}")
        
        print(f"\n3. MODEL ASSESSMENT:")
        print(f"   Data range: [{data_flat.min():.3f}, {data_flat.max():.3f}]")
        print(f"   Model range: [{model_samples.min():.3f}, {model_samples.max():.3f}]")
        
        # Calculate basic fit metrics
        data_mean = np.mean(data_flat)
        model_mean = np.mean(model_samples)
        data_std = np.std(data_flat)
        model_std = np.std(model_samples)
        
        print(f"   Data mean: {data_mean:.3f}, Model mean: {model_mean:.3f}")
        print(f"   Data std: {data_std:.3f}, Model std: {model_std:.3f}")
        
        print("="*60)
    except Exception as e:
        print(f"Error generating HMM-specific diagnostics: {e}")
        print("Please ensure the model has been fit and idata is not None.")

def plot_comparison(
    historical_monthly_data: np.ndarray,
    synthetic_data: np.ndarray,
    output_dir: Path,
    site_column: Optional[int] = 0
) -> None:
    """
    Plot monthly and yearly comparison between historical and synthetic data for the outflow site.
    
    Parameters
    ----------
    historical_monthly_data : np.ndarray
        Historical streamflow data with shape (n_months, n_sites) or (n_months,) for the outflow site
    synthetic_data : np.ndarray
        Synthetic streamflow data with shape (n_months, n_sites, n_ensembles)
    output_dir : Path
        Directory to save plots
    site_column : Optional[int], default=0
        Index of the site to use for comparison. If None, uses the first site.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import random

    # Select outflow site from synthetic data
    if site_column is None:
        site_column = 0
    # synthetic_data: (n_months, n_sites, n_ensembles)
    # Reshape to (n_ensembles, n_months) for the selected site
    syn = synthetic_data[:, :, site_column]  # (n_ensembles, n_months)
    # historical_monthly_data: (n_months, n_sites) or (n_months,)

    hist = historical_monthly_data.iloc[:, site_column]

    n_months = min(syn.shape[1], hist.shape[0])
    months = np.arange(n_months)

    # Compute statistics for synthetic data
    syn_mean = np.mean(syn[:, :n_months], axis=0)
    syn_median = np.median(syn[:, :n_months], axis=0)
    syn_q25 = np.percentile(syn[:, :n_months], 25, axis=0)
    syn_q75 = np.percentile(syn[:, :n_months], 75, axis=0)

    # Select 5 random synthetic ensemble members
    rng = np.random.default_rng(42)
    n_syn_ens = syn.shape[0]
    random_ens = rng.choice(n_syn_ens, size=min(5, n_syn_ens), replace=False)

    # Create figure for monthly comparison
    fig, ax1 = plt.subplots(figsize=(15, 6))

    # Plot historical data
    ax1.plot(months, hist[:n_months], 'k-', label='Historical', alpha=0.8, linewidth=2)

    # Plot synthetic data stats
    # ax1.plot(months, syn_mean, 'r-', label='Synthetic Mean', alpha=0.7)
    ax1.plot(months, syn_median, 'r--', label='Synthetic Median', alpha=0.7)
    ax1.fill_between(months, syn_q25, syn_q75, color='r', alpha=0.2, label='Synthetic IQR')

    # Plot 5 random synthetic ensemble members
    # for idx in random_ens:
    #     ax1.plot(months, syn[idx, :n_months], color='orange', alpha=0.5, lw=1, label='Synthetic Member' if idx == random_ens[0] else None)

    ax1.set_title('Monthly Streamflow Comparison (Outflow Site)', fontsize=14)
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Streamflow', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Monthly comparison for first year (first 12 months) ---
    n_months_plot = min(12, n_months)
    syn_plot = syn[:, :n_months_plot]
    hist_plot = hist[:n_months_plot]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:n_months_plot]

    fig4, ax4 = plt.subplots(figsize=(15, 6))
    # Plot each synthetic ensemble as a line
    for i in random_ens:
        ax4.plot(month_labels, syn_plot[i], color='orange', alpha=0.3, linewidth=1)
    # Plot mean of synthetic ensembles
    ax4.plot(month_labels, syn_plot.mean(axis=0), color='red', linewidth=2, label='Synthetic Mean')
    # Plot historical as a bold line
    ax4.plot(month_labels, hist_plot, color='black', linewidth=2, label='Historical')
    ax4.set_title('Monthly Streamflow Comparison (First Year, Outflow Site)', fontsize=14)
    ax4.set_xlabel('Month', fontsize=12)
    ax4.set_ylabel('Streamflow', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    fig4.savefig(output_dir / 'monthly_firstyear_line_comparison.png')
    plt.close(fig4)

    # Yearly comparison
    def aggregate_yearly(data, n_months):
        n_years = n_months // 12
        data = data[:n_years*12]
        return data.reshape(-1, 12).sum(axis=1)

    syn_yearly = np.array([aggregate_yearly(s, n_months) for s in syn])
    hist_yearly = aggregate_yearly(hist.values, n_months)
    n_years = syn_yearly.shape[1]
    years = np.arange(n_years)

    # Compute statistics for yearly data
    syn_year_mean = np.mean(syn_yearly, axis=0)
    syn_year_median = np.median(syn_yearly, axis=0)
    syn_year_q25 = np.percentile(syn_yearly, 25, axis=0)
    syn_year_q75 = np.percentile(syn_yearly, 75, axis=0)

    # Yearly plot
    fig3, ax3 = plt.subplots(figsize=(15, 6))
    ax3.plot(years, hist_yearly, 'k-', label='Historical', alpha=0.8, linewidth=2)
    # ax3.plot(years, syn_year_mean, 'r-', label='Synthetic Mean', alpha=0.7)
    ax3.plot(years, syn_year_median, 'r--', label='Synthetic Median', alpha=0.7)
    ax3.fill_between(years, syn_year_q25, syn_year_q75, color='r', alpha=0.2, label='Synthetic IQR')
    # for idx in random_ens:
    #     ax3.plot(years, syn_yearly[idx, :], color='orange', alpha=0.5, lw=1, label='Synthetic Member' if idx == random_ens[0] else None)
    # Add horizontal lines for historical mean and median
    hist_mean = np.mean(hist_yearly)
    hist_median = np.median(hist_yearly)
    # ax3.axhline(y=hist_mean, color='black', linestyle=':', linewidth=2, label=f'Historical Mean ({hist_mean:.0f})')
    ax3.axhline(y=hist_median, color='black', linestyle='-.', linewidth=2, label=f'Historical Median ({hist_median:.0f})')
    ax3.set_title('Yearly Streamflow Comparison (Outflow Site)', fontsize=14)
    ax3.set_xlabel('Year', fontsize=12)
    ax3.set_ylabel('Streamflow', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3.savefig(output_dir / 'yearly_comparison.png')
    plt.close(fig3)

    # --- Boxplot of monthly values across years and ensembles (synthetic vs historical) ---
    # syn: (n_ensembles, n_months)
    # hist: pandas Series or array of shape (n_months,)
    n_total_months = syn.shape[1]
    n_ensembles = syn.shape[0]
    n_years = n_total_months // 12
    # Reshape synthetic to (n_ensembles, n_years, 12)
    syn_reshaped = syn[:, :n_years*12].reshape(n_ensembles, n_years, 12)
    synthetic_monthly_box = [syn_reshaped[:, :, m].flatten() for m in range(12)]
    # Reshape historical to (n_years, 12)
    if hasattr(hist, 'values'):
        hist_vals = hist.values
    else:
        hist_vals = np.asarray(hist)
    hist_reshaped = hist_vals[:n_years*12].reshape(n_years, 12)
    historical_monthly_box = [hist_reshaped[:, m].flatten() for m in range(12)]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig_box, ax_box = plt.subplots(figsize=(15, 6))
    positions_hist = np.arange(12) - 0.2
    positions_syn = np.arange(12) + 0.2
    box_hist = ax_box.boxplot(historical_monthly_box, positions=positions_hist, widths=0.3, patch_artist=True,
                              boxprops=dict(facecolor='black', alpha=0.5), showfliers=False)
    box_syn = ax_box.boxplot(synthetic_monthly_box, positions=positions_syn, widths=0.3, patch_artist=True,
                             boxprops=dict(facecolor='orange', alpha=0.5), showfliers=False)
    ax_box.set_xticks(np.arange(12))
    ax_box.set_xticklabels(month_labels)
    ax_box.set_title('Monthly Streamflow Distribution by Month (All Years)', fontsize=14)
    ax_box.set_xlabel('Month', fontsize=12)
    ax_box.set_ylabel('Streamflow', fontsize=12)
    ax_box.legend([box_hist["boxes"][0], box_syn["boxes"][0]], ['Historical', 'Synthetic'])
    ax_box.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_box.savefig(output_dir / 'monthly_boxplot_hist_vs_synth_all_years.png')
    plt.close(fig_box)

    # --- Boxplot of monthly proportion of annual streamflow (synthetic vs historical) ---
    # syn: (n_ensembles, n_months)
    # hist: pandas Series or array of shape (n_months,)
    n_total_months = syn.shape[1]
    n_ensembles = syn.shape[0]
    n_years = n_total_months // 12
    # Reshape synthetic to (n_ensembles, n_years, 12)
    syn_reshaped = syn[:, :n_years*12].reshape(n_ensembles, n_years, 12)
    # Compute monthly proportions for synthetic data
    syn_annual = syn_reshaped.sum(axis=2, keepdims=True)  # (n_ensembles, n_years, 1)
    syn_prop = np.divide(syn_reshaped, syn_annual, out=np.zeros_like(syn_reshaped), where=syn_annual!=0)
    synthetic_monthly_box = [syn_prop[:, :, m].flatten() for m in range(12)]
    # Reshape historical to (n_years, 12)
    if hasattr(hist, 'values'):
        hist_vals = hist.values
    else:
        hist_vals = np.asarray(hist)
    hist_reshaped = hist_vals[:n_years*12].reshape(n_years, 12)
    # Compute monthly proportions for historical data
    hist_annual = hist_reshaped.sum(axis=1, keepdims=True)  # (n_years, 1)
    hist_prop = np.divide(hist_reshaped, hist_annual, out=np.zeros_like(hist_reshaped), where=hist_annual!=0)
    historical_monthly_box = [hist_prop[:, m].flatten() for m in range(12)]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig_box, ax_box = plt.subplots(figsize=(15, 6))
    positions_hist = np.arange(12) - 0.2
    positions_syn = np.arange(12) + 0.2
    box_hist = ax_box.boxplot(historical_monthly_box, positions=positions_hist, widths=0.3, patch_artist=True,
                              boxprops=dict(facecolor='black', alpha=0.5), showfliers=False)
    box_syn = ax_box.boxplot(synthetic_monthly_box, positions=positions_syn, widths=0.3, patch_artist=True,
                             boxprops=dict(facecolor='orange', alpha=0.5), showfliers=False)
    ax_box.set_xticks(np.arange(12))
    ax_box.set_xticklabels(month_labels)
    ax_box.set_title('Monthly Proportion of Annual Streamflow by Month (All Years)', fontsize=14)
    ax_box.set_xlabel('Month', fontsize=12)
    ax_box.set_ylabel('Proportion of Annual Streamflow', fontsize=12)
    ax_box.legend([box_hist["boxes"][0], box_syn["boxes"][0]], ['Historical', 'Synthetic'])
    ax_box.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_box.savefig(output_dir / 'monthly_boxplot_hist_vs_synth_proportion_all_years.png')
    plt.close(fig_box)

"""
Minimal example demonstrating BaGGLS usage on simulated data.

This script generates synthetic data with known ground truth effects,
fits the BaGGLS model, and evaluates performance using RMSE.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.baggls import BaGGLS
from scipy.stats import gamma, bernoulli, norm
import pandas as pd


def generate_simulation_data(n=500, d=10, true_effects_indices=[1, 2, 11], 
                             true_effects_values=[0.8, 0.8, -1.0], seed=42):
    """
    Generate synthetic data for simulation study.
    
    Parameters:
        n (int): Number of observations (default: 500).
        d (int): Number of motifs/features (default: 10).
        true_effects_indices (list): Indices of true non-zero effects.
        true_effects_values (list): Values of true non-zero effects.
        seed (int): Random seed for reproducibility (default: 42).
    
    Returns:
        tuple: (motifs, y, beta_true, design_matrix)
    """
    np.random.seed(seed)
    
    # Generate motif data from gamma distribution
    motifs = pd.DataFrame(
        gamma(a=1).rvs((n, d)),
        columns=[f"m{j+1}" for j in range(d)]
    )
    
    # Build design matrix using BaGGLS (with all effects included)
    baggls_temp = BaGGLS(np.zeros(n), motifs, cut_quantile=1)
    design_matrix = baggls_temp.design_matrix
    p = design_matrix.shape[1]
    
    # Set true coefficients
    beta_true = np.zeros(p)
    beta_true[true_effects_indices] = true_effects_values
    
    # Generate binary response using probit model
    latent = np.matmul(design_matrix, beta_true)
    probabilities = norm.cdf(latent)
    y = bernoulli.rvs(p=probabilities)
    
    return motifs, y, beta_true, design_matrix


def fit_baggls_model(y, motifs, cut_quantile=1, verbose=False):
    """
    Fit BaGGLS model to data.
    
    Parameters:
        y (np.ndarray): Binary response vector.
        motifs (pd.DataFrame): Motif attribution scores.
        verbose (bool): Whether to print convergence information.
    
    Returns:
        BaGGLS: Fitted model object.
    """
    baggls = BaGGLS(y, motifs, cut_quantile=cut_quantile)
    baggls.fit_vb(verbose=verbose)
    return baggls


def compute_posterior_mean(baggls):
    """
    Compute posterior mean of beta coefficients.
    
    Parameters:
        baggls (BaGGLS): Fitted BaGGLS model.
    
    Returns:
        np.ndarray: Posterior mean estimates of beta.
    """
    exp_z = baggls.mu_z + np.sqrt(baggls.sigma2_z) * (2 * baggls.y - 1) * \
            norm.pdf(baggls.mu_z / np.sqrt(baggls.sigma2_z)) / \
            norm.cdf((2 * baggls.y - 1) * baggls.mu_z / np.sqrt(baggls.sigma2_z))
    beta_hat = np.matmul(baggls.b_beta, exp_z)
    return beta_hat


def evaluate_rmse(beta_hat, beta_true):
    """
    Calculate root mean squared error between estimated and true coefficients.
    
    Parameters:
        beta_hat (np.ndarray): Estimated coefficients.
        beta_true (np.ndarray): True coefficients.
    
    Returns:
        float: RMSE value.
    """
    return np.sqrt(np.sum((beta_hat - beta_true)**2))


def run_simulation(n=500, d=10, verbose=False, seed=42):
    """
    Run a complete simulation experiment.
    
    Parameters:
        n (int): Number of observations.
        d (int): Number of motifs.
        verbose (bool): Whether to print detailed output.
        seed (int): Random seed.
    
    Returns:
        dict: Results dictionary with RMSE and other metrics.
    """
    print(f"Running simulation with n={n}, d={d}")
    
    # Generate data
    motifs, y, beta_true, design_matrix = generate_simulation_data(n, d, seed=seed)
    print(f"Generated data: {y.sum()} positive samples out of {n}")
    
    # Fit model
    print("Fitting BaGGLS model...")
    baggls = fit_baggls_model(y, motifs, verbose=verbose)
    
    # Compute estimates
    beta_hat = compute_posterior_mean(baggls)
    
    # Evaluate
    rmse = evaluate_rmse(beta_hat, beta_true)
    print(f"\nResults:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  True non-zero effects: {np.sum(beta_true != 0)}")
    print(f"  Estimated effects (|beta| > 0.1): {np.sum(np.abs(beta_hat) > 0.1)}")
    
    return {
        'rmse': rmse,
        'beta_hat': beta_hat,
        'beta_true': beta_true,
        'model': baggls
    }

''' Uncomment to run simulation example with the script. Otherwise, use the Jupyter notebook simulation_paper.ipynb.
if __name__ == "__main__":
    """
    Run simulation example.
    
    Usage:
        python src/simulation/simulation.py
    """
    results = run_simulation(n=500, d=10, verbose=False, seed=42)
    
    # Optionally print top effects
    print("\nTop 5 absolute effects:")
    top_effects, top_values = results['model'].absolute_effects(top_n=5, verbose=True)
'''
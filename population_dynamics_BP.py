# code kindly provided by Jerome Garnier-Brun

"""
Population dynamics for the BP distributional fixed point
on a binary symmetric broadcasting tree.

Saves binned distributions and summary statistics to ./sim_data/
"""

import numpy as np
from multiprocessing import Pool, cpu_count


def bp_update(m1, m2, theta):
    """Vectorized BP magnetization update."""
    return theta * (m1 + m2) / (1.0 + theta**2 * m1 * m2)


def population_dynamics(theta, M=1000, n_sweeps=150, seed=42):
    """
    Run population dynamics conditioned on parent x = +1.

    Each child is +1 with prob (1+theta)/2, otherwise -1 (flip message sign).

    Returns the converged pool of M magnetization samples.
    """
    rng = np.random.default_rng(seed)

    # Initialize with small positive bias
    pool = rng.uniform(0.0, 0.1, size=M)
    p_plus = (1.0 + theta) / 2.0

    for _ in range(n_sweeps):
        idx1 = rng.integers(0, M, size=M)
        idx2 = rng.integers(0, M, size=M)
        m1 = pool[idx1].copy()
        m2 = pool[idx2].copy()

        flip1 = rng.random(M) > p_plus
        flip2 = rng.random(M) > p_plus
        m1[flip1] *= -1.0
        m2[flip2] *= -1.0

        pool[:] = bp_update(m1, m2, theta)

    return pool


def run_single(args):
    """Wrapper for starmap: (index, epsilon, M, n_sweeps, seed) -> results."""
    idx, eps, M, n_sweeps, seed, bin_edges = args
    theta = 1.0 - 2.0 * eps
    pool = population_dynamics(theta, M=M, n_sweeps=n_sweeps, seed=seed)

    # Summary statistics
    p_error = np.mean(pool < 0)
    mean_m = np.mean(pool)
    var_m = np.var(pool)

    # Binned distribution
    hist, _ = np.histogram(pool, bins=bin_edges, density=True)

    return idx, p_error, mean_m, var_m, hist


if __name__ == "__main__":
    # --- Parameters ---
    n_eps = 50
    eps_values = np.linspace(0.001, 0.3, n_eps)
    M = int(1e6)
    n_sweeps = 150
    base_seed = 42

    # Shared bin edges for all histograms
    n_bins = 500
    bin_edges = np.linspace(-1.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Pre-allocate output arrays
    errors = np.empty(n_eps)
    means = np.empty(n_eps)
    variances = np.empty(n_eps)
    histograms = np.empty((n_eps, n_bins))

    # Build argument list (one seed per epsilon for reproducibility)
    args = [
        (i, eps, M, n_sweeps, base_seed + i, bin_edges)
        for i, eps in enumerate(eps_values)
    ]

    # --- Parallel execution ---
    n_workers = cpu_count()
    print(f"Running {n_eps} epsilon values on {n_workers} workers, M={M}, sweeps={n_sweeps}")

    with Pool(n_workers) as p:
        results = p.map(run_single, args)

    # Unpack into pre-allocated arrays
    for idx, p_error, mean_m, var_m, hist in results:
        errors[idx] = p_error
        means[idx] = mean_m
        variances[idx] = var_m
        histograms[idx] = hist

    # --- Save ---
    np.savez(
        f"./sim_data/BP_pop_dyn_results_M={M}_sweeps={n_sweeps}.npz",
        eps=eps_values,
        errors=errors,
        means=means,
        variances=variances,
        histograms=histograms,
        bin_centers=bin_centers,
        bin_edges=bin_edges,
    )
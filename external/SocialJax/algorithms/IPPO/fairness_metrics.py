"""
Fairness metrics for multi-agent reinforcement learning evaluation.
Implements exact formulas from the paper (α-fairness, Jain index, Gini coefficient).

Note: Paper assumes r_i > 0 for GeoMean and non-zero sums for Gini/Jain.
We handle edge cases (negative returns, zeros) by shifting to positive.
"""
import jax.numpy as jnp


def gini_coefficient(x):
    """
    Gini coefficient as defined in the paper.
    
    Formula: G = (2 * Σ(i * r_(i))) / (n * Σ(r_i)) - (n+1)/n
    where r_(i) are sorted in increasing order.
    
    Paper assumption: Σ(r_i) > 0
    We handle negatives by shifting to positive (preserves relative inequality).
    """
    x = jnp.array(x)
    n = len(x)
    
    # Edge cases
    all_equal = jnp.all(x == x[0]) if n > 0 else True
    is_edge = (n <= 1) | all_equal
    
    # Shift to non-negative (paper assumes positive, but we handle negatives)
    min_x = jnp.min(x)
    x_shifted = x - min_x + 1e-10
    
    # Paper formula: G = (2 * Σ(i * r_(i))) / (n * Σ(r_i)) - (n+1)/n
    x_sorted = jnp.sort(x_shifted)
    index = jnp.arange(1, n + 1, dtype=jnp.float32)
    numerator = 2 * jnp.sum(index * x_sorted)
    denominator = n * jnp.sum(x_shifted)
    
    gini = jnp.where(is_edge | (denominator <= 1e-10),
                     0.0,
                     numerator / denominator - (n + 1) / n)
    
    return jnp.clip(gini, 0.0, 1.0)


def jain_index(x):
    """
    Jain's Fairness Index as defined in the paper.
    
    Formula: J = (Σ(r_i))^2 / (n * Σ(r_i^2))
    
    Paper assumption: Σ(r_i^2) > 0
    Range: 1/n ≤ J ≤ 1 (higher = more equal)
    """
    x = jnp.array(x)
    n = len(x)
    
    # Edge cases
    all_equal = jnp.all(x == x[0]) if n > 0 else True
    is_edge = (n <= 1) | all_equal
    
    # Shift to non-negative
    min_x = jnp.min(x)
    x_shifted = x - min_x + 1e-10
    
    # Paper formula: J = (Σ(r_i))^2 / (n * Σ(r_i^2))
    sum_x = jnp.sum(x_shifted)
    sum_x_sq = jnp.sum(x_shifted ** 2)
    
    jain = jnp.where(is_edge | (sum_x_sq <= 1e-10),
                     1.0,
                     sum_x ** 2 / (n * sum_x_sq))
    
    return jnp.clip(jain, 1.0 / n if n > 0 else 1.0, 1.0)


def compute_fairness_metrics(returns):
    """
    Compute all fairness metrics from agent returns using paper formulas.
    
    Returns:
        Dictionary with:
        - mean_return: (1/n) * Σ(r_i) [α=0]
        - geomean_return: exp((1/n) * Σ(log(r_i))) [α=1, Nash Social Welfare]
        - min_return: min_i(r_i) [α→∞]
        - gini_coefficient: Gini coefficient
        - jain_index: Jain's fairness index
        - max_return: max_i(r_i)
    """
    returns = jnp.array(returns)
    n = len(returns)
    
    # Mean return (α=0): (1/n) * Σ(r_i)
    mean_return = jnp.mean(returns)
    
    # Geometric Mean / Nash Social Welfare (α=1): exp((1/n) * Σ(log(r_i)))
    # Paper assumes r_i > 0, but we handle negatives by shifting
    min_ret = jnp.min(returns)
    eps = 1e-6
    shifted_returns = returns - min_ret + eps
    geomean_return = jnp.exp(jnp.mean(jnp.log(shifted_returns)))
    
    # Minimum return (α→∞): min_i(r_i)
    min_return = jnp.min(returns)
    
    # Maximum return
    max_return = jnp.max(returns)
    
    return {
        "mean_return": mean_return,
        "geomean_return": geomean_return,
        "min_return": min_return,
        "max_return": max_return,
        "gini_coefficient": gini_coefficient(returns),
        "jain_index": jain_index(returns),
    }

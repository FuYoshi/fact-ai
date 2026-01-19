"""
Fairness Metrics for Multi-Agent Reinforcement Learning

Implements metrics from the paper:
- α-fairness (Mean, GeoMean, Min)
- Gini coefficient 
- Jain's fairness index

HANDLING NEGATIVE RETURNS
=========================
Gini, Jain, and GeoMean require positive values.
When any agent has non-positive returns, we SKIP those agents
for these metrics and set a flag.

Simple metrics (mean, min, max, gap) always use ALL agents.

NOTE: This implementation is JIT-compatible (no boolean indexing).
We use jnp.where to mask out non-positive values.
"""
import jax.numpy as jnp
from typing import Dict

_EPS = 1e-8


def gini_coefficient(returns: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Gini coefficient on POSITIVE returns only.
    
    Formula: G = Σ_i Σ_j |r_i - r_j| / (2N * Σ r_i)
    
    Agents with returns <= 0 are skipped via masking.
    Returns NaN if fewer than 2 positive agents.
    """
    returns = jnp.asarray(returns, dtype=jnp.float32)
    n_total = returns.size
    
    # Mask for positive values
    positive_mask = returns > _EPS
    n_positive = jnp.sum(positive_mask)
    
    # Replace non-positive with 0 for computation (they won't contribute)
    masked_returns = jnp.where(positive_mask, returns, 0.0)
    
    # Compute pairwise differences (only positive pairs contribute)
    # |r_i - r_j| but only when both i and j are positive
    mask_matrix = positive_mask[:, None] & positive_mask[None, :]
    diff_matrix = jnp.abs(masked_returns[:, None] - masked_returns[None, :])
    masked_diff = jnp.where(mask_matrix, diff_matrix, 0.0)
    numerator = jnp.sum(masked_diff)
    
    # Denominator uses only positive values
    sum_positive = jnp.sum(masked_returns)
    denominator = 2.0 * n_positive * sum_positive
    
    # Check if all positive values are equal
    max_val = jnp.max(masked_returns)
    min_positive = jnp.where(positive_mask, returns, jnp.inf).min()
    all_equal = (max_val - min_positive) < _EPS
    
    # Compute Gini with safety checks
    gini = jnp.where(
        n_positive < 2,
        jnp.nan,
        jnp.where(
            all_equal | (denominator <= _EPS),
            0.0,
            numerator / denominator
        )
    )
    
    return jnp.clip(gini, 0.0, 1.0)


def jain_index(returns: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Jain's fairness index on POSITIVE returns only.
    
    Formula: J = (Σ r_i)² / (N * Σ r_i²)
    
    Agents with returns <= 0 are skipped via masking.
    Returns NaN if fewer than 2 positive agents.
    """
    returns = jnp.asarray(returns, dtype=jnp.float32)
    
    # Mask for positive values
    positive_mask = returns > _EPS
    n_positive = jnp.sum(positive_mask)
    
    # Replace non-positive with 0
    masked_returns = jnp.where(positive_mask, returns, 0.0)
    
    # Compute sums over positive values only
    sum_r = jnp.sum(masked_returns)
    sum_r_sq = jnp.sum(masked_returns ** 2)
    
    numerator = sum_r ** 2
    denominator = n_positive * sum_r_sq
    
    # Check if all positive values are equal
    max_val = jnp.max(masked_returns)
    min_positive = jnp.where(positive_mask, returns, jnp.inf).min()
    all_equal = (max_val - min_positive) < _EPS
    
    # Compute Jain with safety checks
    jain = jnp.where(
        n_positive < 2,
        jnp.nan,
        jnp.where(
            all_equal | (denominator <= _EPS),
            1.0,
            numerator / denominator
        )
    )
    
    # Clip to valid range [1/n, 1]
    lower_bound = jnp.where(n_positive > 0, 1.0 / n_positive, 0.0)
    return jnp.clip(jain, lower_bound, 1.0)


def geometric_mean(returns: jnp.ndarray) -> jnp.ndarray:
    """
    Compute geometric mean on POSITIVE returns only.
    
    Agents with returns <= 0 are skipped via masking.
    Returns NaN if no positive agents.
    """
    returns = jnp.asarray(returns, dtype=jnp.float32)
    
    # Mask for positive values
    positive_mask = returns > _EPS
    n_positive = jnp.sum(positive_mask)
    
    # Replace non-positive with 1.0 for log (log(1)=0, won't affect sum)
    safe_returns = jnp.where(positive_mask, returns, 1.0)
    
    # Sum of logs, but only count positive values
    log_sum = jnp.sum(jnp.where(positive_mask, jnp.log(safe_returns), 0.0))
    
    # Geometric mean = exp(mean of logs)
    geomean = jnp.where(
        n_positive < 1,
        jnp.nan,
        jnp.exp(log_sum / n_positive)
    )
    
    return geomean


def compute_fairness_metrics(returns: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Compute all fairness metrics from per-agent returns.
    
    Args:
        returns: Per-agent returns, shape (num_agents,)
    
    Returns:
        Dictionary with:
        
        ALWAYS COMPUTED ON ALL AGENTS:
        - mean_return: Arithmetic mean
        - min_return: Worst-off agent
        - max_return: Best-off agent  
        - return_gap: max - min
        
        COMPUTED ON POSITIVE AGENTS ONLY (NaN if <2 positive):
        - geomean_return: Geometric mean
        - gini_coefficient: Gini index
        - jain_index: Jain index
        
        METADATA:
        - num_positive_agents: How many agents had positive returns
        - num_agents: Total number of agents
    """
    returns = jnp.asarray(returns, dtype=jnp.float32)
    n_total = returns.size
    n_positive = jnp.sum(returns > _EPS)
    
    # Simple metrics - always use ALL agents
    mean_return = jnp.mean(returns)
    min_return = jnp.min(returns)
    max_return = jnp.max(returns)
    return_gap = max_return - min_return
    
    # Complex metrics - only positive agents
    geomean_return = geometric_mean(returns)
    gini = gini_coefficient(returns)
    jain = jain_index(returns)
    
    return {
        # Always valid (all agents)
        "mean_return": mean_return,
        "min_return": min_return,
        "max_return": max_return,
        "return_gap": return_gap,
        # Only positive agents (may be NaN)
        "geomean_return": geomean_return,
        "gini_coefficient": gini,
        "jain_index": jain,
        # Metadata
        "num_positive_agents": n_positive,
        "num_agents": n_total,
    }


# =============================================================================
# Testing
# =============================================================================

def _test_metrics():
    print("=" * 70)
    print("FAIRNESS METRICS TEST (Skip Negative Strategy)")
    print("=" * 70)
    
    test_cases = [
        ("Equal [50, 50]", [50.0, 50.0]),
        ("2:1 ratio [100, 50]", [100.0, 50.0]),
        ("10:1 ratio [100, 10]", [100.0, 10.0]),
        ("One zero [100, 0]", [100.0, 0.0]),
        ("One negative [100, -50]", [100.0, -50.0]),
        ("Ind-like [100, -100]", [100.0, -100.0]),
        ("Both negative [-50, -100]", [-50.0, -100.0]),
        ("3 agents [100, 50, -20]", [100.0, 50.0, -20.0]),
        ("Paper Col ~37:1", [100.0, 2.67]),
        ("FCGrad fair [60, 60]", [60.0, 60.0]),
    ]
    
    for name, vals in test_cases:
        returns = jnp.array(vals)
        m = compute_fairness_metrics(returns)
        
        n_pos = int(m['num_positive_agents'])
        n_tot = int(m['num_agents'])
        
        print(f"\n{name}")
        print(f"  All agents:      Mean={m['mean_return']:.1f}, Min={m['min_return']:.1f}, "
              f"Max={m['max_return']:.1f}, Gap={m['return_gap']:.1f}")
        
        if n_pos >= 2:
            print(f"  Positive only:   Gini={m['gini_coefficient']:.4f}, "
                  f"Jain={m['jain_index']:.4f}, GeoMean={m['geomean_return']:.2f}")
            print(f"                   ({n_pos}/{n_tot} agents used)")
        elif n_pos == 1:
            print(f"  Positive only:   Only 1 positive agent → Gini/Jain = NaN")
        else:
            print(f"  Positive only:   No positive agents → Gini/Jain = NaN")
    
    print("\n" + "=" * 70)
    print("KEY POINTS:")
    print("  - Mean/Min/Max/Gap always use ALL agents")
    print("  - Gini/Jain/GeoMean only use POSITIVE agents")
    print("  - If <2 positive agents, Gini/Jain return NaN")
    print("=" * 70)


if __name__ == "__main__":
    _test_metrics()
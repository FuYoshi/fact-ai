"""
Fairness Metrics for Multi-Agent Reinforcement Learning

Implements metrics from the paper:
- α-fairness (Mean, GeoMean, Min)
- Gini coefficient
- Jain's fairness index

HANDLING NEGATIVE RETURNS (SHIFT STRATEGY)
=========================================
Gini, Jain, and GeoMean require strictly positive values.

Instead of SKIPPING agents with non-positive returns, we SHIFT the whole
return vector to become strictly positive:

    r_shifted = r - min(r) + delta    if min(r) <= 0
    r_shifted = r                     otherwise

This avoids NaNs for cases like Ind: [100, -100].

Simple metrics (mean, min, max, gap) are computed on RAW returns (all agents).
Fairness metrics (Gini, Jain, GeoMean) are computed on SHIFTED returns.

NOTE: This implementation is JIT-compatible (no boolean indexing).
We only use jnp.where, no Python loops in the metric functions.
"""

import jax.numpy as jnp
from typing import Dict

_EPS = 1e-8


def shift_to_positive(returns: jnp.ndarray, delta: float = 1e-3) -> jnp.ndarray:
    """
    Shift returns so all values are strictly positive.

    If min(returns) <= 0:
        r' = r - min(r) + delta
    else:
        r' = r

    Args:
        returns: array of shape (num_agents,)
        delta: small constant to ensure strictly positive minimum

    Returns:
        shifted_returns: strictly positive array
    """
    returns = jnp.asarray(returns, dtype=jnp.float32)
    min_r = jnp.min(returns)
    shift = jnp.where(min_r <= 0.0, -min_r + delta, 0.0)
    shifted = returns + shift

    # Safety: avoid exact zeros due to numerical issues
    return jnp.maximum(shifted, _EPS)


def gini_coefficient(positive_returns: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Gini coefficient assuming STRICTLY POSITIVE returns.

    Formula:
        G = Σ_i Σ_j |r_i - r_j| / (2N * Σ r_i)

    Returns NaN if fewer than 2 agents.
    """
    r = jnp.asarray(positive_returns, dtype=jnp.float32)
    n = r.size

    # Pairwise absolute differences
    diff = jnp.abs(r[:, None] - r[None, :])
    numerator = jnp.sum(diff)
    denominator = 2.0 * n * jnp.sum(r)

    # If all values equal -> Gini = 0
    all_equal = (jnp.max(r) - jnp.min(r)) < _EPS

    gini = jnp.where(
        n < 2,
        jnp.nan,
        jnp.where(
            all_equal | (denominator <= _EPS),
            0.0,
            numerator / denominator
        )
    )
    return jnp.clip(gini, 0.0, 1.0)


def jain_index(positive_returns: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Jain's fairness index assuming STRICTLY POSITIVE returns.

    Formula:
        J = (Σ r_i)² / (N * Σ r_i²)

    Returns NaN if fewer than 2 agents.
    """
    r = jnp.asarray(positive_returns, dtype=jnp.float32)
    n = r.size

    sum_r = jnp.sum(r)
    sum_r_sq = jnp.sum(r ** 2)

    numerator = sum_r ** 2
    denominator = n * sum_r_sq

    all_equal = (jnp.max(r) - jnp.min(r)) < _EPS

    jain = jnp.where(
        n < 2,
        jnp.nan,
        jnp.where(
            all_equal | (denominator <= _EPS),
            1.0,
            numerator / denominator
        )
    )

    # Clip to valid range [1/n, 1]
    return jnp.clip(jain, 1.0 / jnp.maximum(n, 1), 1.0)


def geometric_mean(positive_returns: jnp.ndarray) -> jnp.ndarray:
    """
    Compute geometric mean assuming STRICTLY POSITIVE returns.

    GeoMean = exp( mean( log(r_i) ) )

    Returns NaN if no agents.
    """
    r = jnp.asarray(positive_returns, dtype=jnp.float32)
    n = r.size

    # Extra safety: clamp away from 0
    r = jnp.maximum(r, _EPS)

    log_sum = jnp.sum(jnp.log(r))
    geomean = jnp.where(n < 1, jnp.nan, jnp.exp(log_sum / n))
    return geomean


def compute_fairness_metrics(
    returns: jnp.ndarray,
    shift_delta: float = 1e-3
) -> Dict[str, jnp.ndarray]:
    """
    Compute all fairness metrics from per-agent returns.

    Args:
        returns: Per-agent returns, shape (num_agents,)
        shift_delta: delta used for shifting (ensures strictly positive)

    Returns:
        Dictionary with:

        RAW (always on all agents):
        - mean_return
        - min_return
        - max_return
        - return_gap

        FAIRNESS (computed on shifted returns):
        - geomean_return
        - gini_coefficient
        - jain_index

        METADATA:
        - num_agents
        - shift_amount (0 if no shift applied)
        - shifted_min (should be ~delta or >0)
    """
    returns = jnp.asarray(returns, dtype=jnp.float32)
    n_total = returns.size

    # Raw metrics (always meaningful)
    mean_return = jnp.mean(returns)
    min_return = jnp.min(returns)
    max_return = jnp.max(returns)
    return_gap = max_return - min_return

    # Shift for fairness metrics
    shifted_returns = shift_to_positive(returns, delta=shift_delta)

    # Fairness metrics on shifted returns
    geomean_return = geometric_mean(shifted_returns)
    gini = gini_coefficient(shifted_returns)
    jain = jain_index(shifted_returns)

    # Metadata about shift
    shift_amount = jnp.min(shifted_returns) - jnp.min(returns)  # equals applied shift (approximately)
    shifted_min = jnp.min(shifted_returns)

    return {
        # Raw (all agents)
        "mean_return": mean_return,
        "min_return": min_return,
        "max_return": max_return,
        "return_gap": return_gap,
        # Fairness (shifted)
        "geomean_return": geomean_return,
        "gini_coefficient": gini,
        "jain_index": jain,
        # Metadata
        "num_agents": n_total,
        "shift_amount": shift_amount,
        "shifted_min": shifted_min,
    }


# =============================================================================
# Testing
# =============================================================================

def _test_metrics():
    print("=" * 70)
    print("FAIRNESS METRICS TEST (Shift Strategy)")
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
        m = compute_fairness_metrics(returns, shift_delta=1e-3)

        n_tot = int(m["num_agents"])

        print(f"\n{name}")
        print(f"  Raw:            Mean={m['mean_return']:.1f}, Min={m['min_return']:.1f}, "
              f"Max={m['max_return']:.1f}, Gap={m['return_gap']:.1f}")
        print(f"  Shifted fairness Gini={m['gini_coefficient']:.4f}, "
              f"Jain={m['jain_index']:.4f}, GeoMean={m['geomean_return']:.2f}")
        print(f"  Shift metadata: shift_amount={float(m['shift_amount']):.4f}, "
              f"shifted_min={float(m['shifted_min']):.4f} ({n_tot}/{n_tot} agents used)")

    print("\n" + "=" * 70)
    print("KEY POINTS:")
    print("  - Mean/Min/Max/Gap use RAW returns (all agents)")
    print("  - Gini/Jain/GeoMean use SHIFTED returns (no skipping, no NaNs for Ind)")
    print("  - Shift rule: r' = r - min(r) + delta if min(r) <= 0")
    print("=" * 70)


if __name__ == "__main__":
    _test_metrics()

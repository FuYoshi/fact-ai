"""
Fairness Metrics for Multi-Agent Reinforcement Learning

Implements metrics from the paper:
- α-fairness (Mean, GeoMean, Min)
- Gini coefficient (paper formula: mean absolute difference)
- Jain's fairness index

Reference formulas from paper (Section 4.1):
- Gini: G = Σ_i Σ_j |r_i - r_j| / (2N * Σ r_i)
- Jain: J = (Σ r_i)² / (N * Σ r_i²)
- α-fairness: U_α = Σ r_i^(1-α) / (1-α) for α ≠ 1, Σ log(r_i) for α = 1

Note on negative returns:
    The paper assumes r_i > 0 for these metrics. When returns can be negative
    (e.g., Ind objective in Coin Game where green agent can get ~-100),
    we shift all values to be positive: r_i' = r_i - min(r) + ε
    This preserves relative differences while making the metrics well-defined.
"""
import jax.numpy as jnp
from typing import Dict

_EPS = 1e-8


def _ensure_positive(x: jnp.ndarray) -> jnp.ndarray:
    """
    Shift values to ensure all are positive.
    
    This is necessary because Gini, Jain, and GeoMean require positive values.
    Shifting preserves relative differences between agents.
    """
    min_val = jnp.min(x)
    # Only shift if minimum is <= 0
    x_shifted = jnp.where(min_val <= _EPS, x - min_val + _EPS, x)
    return x_shifted


def gini_coefficient(returns: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Gini coefficient using the paper's formula.
    
    Formula (from paper, page 9):
        G = Σ_i Σ_j |r_i - r_j| / (2N * Σ r_i)
    
    Args:
        returns: Per-agent returns, shape (num_agents,)
    
    Returns:
        Gini coefficient in [0, 1]. 
        0 = perfect equality, 1 = maximum inequality.
        For N=2 agents, maximum possible Gini is 0.5.
    """
    returns = jnp.asarray(returns, dtype=jnp.float32)
    n = returns.size
    
    # Handle edge cases
    if n == 0:
        return jnp.array(0.0, dtype=jnp.float32)
    if n == 1:
        return jnp.array(0.0, dtype=jnp.float32)
    
    # Check if all values are equal
    all_equal = jnp.allclose(returns, returns[0], rtol=1e-5)
    
    # Shift to positive domain
    returns_pos = _ensure_positive(returns)
    
    # Compute pairwise absolute differences: |r_i - r_j| for all i, j
    # Using broadcasting: (n, 1) - (1, n) -> (n, n)
    diff_matrix = jnp.abs(returns_pos[:, None] - returns_pos[None, :])
    sum_abs_diff = jnp.sum(diff_matrix)
    
    # Compute denominator: 2 * N * Σ r_i
    sum_returns = jnp.sum(returns_pos)
    denominator = 2.0 * n * sum_returns
    
    # Compute Gini
    gini = jnp.where(
        all_equal | (denominator <= _EPS),
        0.0,
        sum_abs_diff / denominator
    )
    
    return jnp.clip(gini, 0.0, 1.0)


def jain_index(returns: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Jain's fairness index.
    
    Formula (from paper, page 9):
        J = (Σ r_i)² / (N * Σ r_i²)
    
    Args:
        returns: Per-agent returns, shape (num_agents,)
    
    Returns:
        Jain index in [1/N, 1].
        1 = perfect equality, 1/N = maximum inequality.
        For N=2 agents, minimum possible Jain is 0.5.
    """
    returns = jnp.asarray(returns, dtype=jnp.float32)
    n = returns.size
    
    # Handle edge cases
    if n == 0:
        return jnp.array(1.0, dtype=jnp.float32)
    if n == 1:
        return jnp.array(1.0, dtype=jnp.float32)
    
    # Check if all values are equal
    all_equal = jnp.allclose(returns, returns[0], rtol=1e-5)
    
    # Shift to positive domain
    returns_pos = _ensure_positive(returns)
    
    # Compute numerator: (Σ r_i)²
    sum_returns = jnp.sum(returns_pos)
    numerator = sum_returns ** 2
    
    # Compute denominator: N * Σ r_i²
    sum_returns_sq = jnp.sum(returns_pos ** 2)
    denominator = n * sum_returns_sq
    
    # Compute Jain index
    jain = jnp.where(
        all_equal | (denominator <= _EPS),
        1.0,
        numerator / denominator
    )
    
    # Jain index is bounded by [1/N, 1]
    lower_bound = 1.0 / n
    return jnp.clip(jain, lower_bound, 1.0)


def geometric_mean(returns: jnp.ndarray) -> jnp.ndarray:
    """
    Compute geometric mean (Nash Social Welfare, α=1 in α-fairness).
    
    Formula:
        GeoMean = exp(Σ log(r_i) / N) = (Π r_i)^(1/N)
    
    This is the α-fairness utility with α=1.
    
    Args:
        returns: Per-agent returns, shape (num_agents,)
    
    Returns:
        Geometric mean of returns.
    """
    returns = jnp.asarray(returns, dtype=jnp.float32)
    n = returns.size
    
    if n == 0:
        return jnp.array(0.0, dtype=jnp.float32)
    
    # Shift to positive domain (log requires positive values)
    returns_pos = _ensure_positive(returns)
    
    # Compute geometric mean: exp(mean(log(r)))
    log_returns = jnp.log(returns_pos)
    geomean = jnp.exp(jnp.mean(log_returns))
    
    return geomean


def alpha_fairness_utility(returns: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """
    Compute α-fairness utility function.
    
    Formula (from paper, Equation 4):
        U_α = Σ r_i^(1-α) / (1-α)  if α ≠ 1
        U_α = Σ log(r_i)           if α = 1
    
    Special cases:
        α = 0: Sum of returns (collective welfare)
        α = 1: Sum of log returns (Nash Social Welfare / log of geometric mean)
        α → ∞: Minimum return (max-min fairness)
    
    Args:
        returns: Per-agent returns, shape (num_agents,)
        alpha: Fairness parameter (higher = more emphasis on fairness)
    
    Returns:
        α-fairness utility value.
    """
    returns = jnp.asarray(returns, dtype=jnp.float32)
    returns_pos = _ensure_positive(returns)
    
    if jnp.abs(alpha - 1.0) < 1e-6:
        # α = 1: log utility (Nash Social Welfare)
        return jnp.sum(jnp.log(returns_pos))
    else:
        # General case
        return jnp.sum(jnp.power(returns_pos, 1.0 - alpha)) / (1.0 - alpha)


def compute_fairness_metrics(returns: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Compute all fairness metrics from per-agent returns.
    
    This is the main function to call for evaluation.
    
    Args:
        returns: Per-agent returns, shape (num_agents,)
    
    Returns:
        Dictionary containing:
            - mean_return: Mean (α=0, collective welfare)
            - geomean_return: Geometric mean (α=1, Nash Social Welfare)
            - min_return: Minimum (α→∞, max-min fairness)
            - max_return: Maximum (for diagnostics)
            - gini_coefficient: Gini (0=equal, higher=unequal)
            - jain_index: Jain (1=equal, lower=unequal)
    """
    returns = jnp.asarray(returns, dtype=jnp.float32)
    
    # Basic statistics (computed on original returns, not shifted)
    mean_return = jnp.mean(returns)
    min_return = jnp.min(returns)
    max_return = jnp.max(returns)
    
    # Geometric mean (requires positive values, so uses shifted returns internally)
    geomean_return = geometric_mean(returns)
    
    # Fairness indices (require positive values, so use shifted returns internally)
    gini = gini_coefficient(returns)
    jain = jain_index(returns)
    
    return {
        "mean_return": mean_return,
        "geomean_return": geomean_return,
        "min_return": min_return,
        "max_return": max_return,
        "gini_coefficient": gini,
        "jain_index": jain,
    }


# =============================================================================
# Testing / Validation
# =============================================================================

def _test_metrics():
    """Test the metrics with known values."""
    print("=" * 60)
    print("Testing Fairness Metrics")
    print("=" * 60)
    
    # Test 1: Perfect equality
    print("\n1. Perfect equality [50, 50]:")
    returns = jnp.array([50.0, 50.0])
    metrics = compute_fairness_metrics(returns)
    print(f"   Gini = {metrics['gini_coefficient']:.4f} (expected: 0)")
    print(f"   Jain = {metrics['jain_index']:.4f} (expected: 1)")
    
    # Test 2: Maximum inequality for N=2 (one has all, other has none)
    print("\n2. Maximum inequality [100, 0] (after shifting to [100+ε, ε]):")
    returns = jnp.array([100.0, 0.0])
    metrics = compute_fairness_metrics(returns)
    print(f"   Gini = {metrics['gini_coefficient']:.4f} (expected: ~0.5)")
    print(f"   Jain = {metrics['jain_index']:.4f} (expected: ~0.5)")
    
    # Test 3: Negative returns (like Ind in Coin Game)
    print("\n3. Negative returns [100, -100] (shifted to [200+ε, ε]):")
    returns = jnp.array([100.0, -100.0])
    metrics = compute_fairness_metrics(returns)
    print(f"   Mean = {metrics['mean_return']:.2f}")
    print(f"   Min = {metrics['min_return']:.2f}")
    print(f"   Gini = {metrics['gini_coefficient']:.4f} (expected: ~0.5)")
    print(f"   Jain = {metrics['jain_index']:.4f} (expected: ~0.5)")
    
    # Test 4: Paper's Col scenario (15:1 ratio approximation)
    print("\n4. Col-like scenario [150, 10] (15:1 ratio):")
    returns = jnp.array([150.0, 10.0])
    metrics = compute_fairness_metrics(returns)
    print(f"   Gini = {metrics['gini_coefficient']:.4f}")
    print(f"   Jain = {metrics['jain_index']:.4f}")
    
    # Test 5: Paper's reported values imply ~37:1 ratio
    print("\n5. Testing ~37:1 ratio [100, 2.67]:")
    returns = jnp.array([100.0, 100.0/37.5])
    metrics = compute_fairness_metrics(returns)
    print(f"   Gini = {metrics['gini_coefficient']:.4f} (paper reports: 0.474)")
    print(f"   Jain = {metrics['jain_index']:.4f} (paper reports: 0.526)")
    
    # Test 6: FCGrad-like fair outcome
    print("\n6. FCGrad-like fair outcome [60, 60]:")
    returns = jnp.array([60.0, 60.0])
    metrics = compute_fairness_metrics(returns)
    print(f"   Gini = {metrics['gini_coefficient']:.4f} (expected: ~0)")
    print(f"   Jain = {metrics['jain_index']:.4f} (expected: ~1)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    _test_metrics()
"""
Fairness metrics for multi-agent reinforcement learning evaluation.
Implements exact formulas from the paper (α-fairness, Jain index, Gini coefficient).

Assumes aggregated episode returns are non-negative (as per paper).
"""
import jax.numpy as jnp
import jax


def gini_coefficient(x):
    """
    Gini coefficient as defined in the paper.
    
    Formula: G = (2 * Σ(i * r_(i))) / (n * Σ(r_i)) - (n+1)/n
    where r_(i) are sorted in increasing order.
    """
    x = jnp.array(x)
    n = len(x)
    
    # Debug: print input values
    jax.debug.print("GINI INPUT: x={x}, n={n}", x=x, n=n)
    
    if n <= 1:
        return 0.0
    
    # Paper formula
    x_sorted = jnp.sort(x)
    index = jnp.arange(1, n + 1, dtype=jnp.float32)
    numerator = 2 * jnp.sum(index * x_sorted)
    denominator = n * jnp.sum(x)
    
    # Debug: print intermediate values
    jax.debug.print("GINI CALC: x_sorted={sorted}, numerator={num}, denominator={den}, sum_x={sumx}", 
                    sorted=x_sorted, num=numerator, den=denominator, sumx=jnp.sum(x))
    
    gini = jnp.where(denominator <= 1e-10,
                     0.0,
                     numerator / denominator - (n + 1) / n)
    
    gini_clipped = jnp.clip(gini, 0.0, 1.0)
    
    # Debug: print final value
    jax.debug.print("GINI RESULT: raw={raw}, clipped={clipped}", raw=gini, clipped=gini_clipped)
    
    return gini_clipped


def jain_index(x):
    """
    Jain's Fairness Index as defined in the paper.
    
    Formula: J = (Σ(r_i))^2 / (n * Σ(r_i^2))
    Range: 1/n ≤ J ≤ 1 (higher = more equal)
    
    Note: 
    - J = 0.5 for 2 agents typically means perfect inequality (one agent has 0 return).
    - However, if one agent has negative returns, J can be < 0.5, which gets clipped to 0.5.
    - Negative returns indicate poor performance and violate paper's assumption of non-negative returns.
    """
    x = jnp.array(x)
    n = len(x)
    
    # Debug: print input values
    jax.debug.print("JAIN INPUT: x={x}, n={n}", x=x, n=n)
    
    if n <= 1:
        return 1.0
    
    # Handle case where all returns are zero
    sum_x_sq = jnp.sum(x ** 2)
    if sum_x_sq <= 1e-10:
        jax.debug.print("JAIN: all returns zero, returning 1.0")
        return 1.0
    
    sum_x = jnp.sum(x)
    jain = sum_x ** 2 / (n * sum_x_sq)
    
    # Debug: print intermediate values
    jax.debug.print("JAIN CALC: sum_x={sumx}, sum_x_sq={sumsq}, jain_raw={raw}, min_bound={minb}", 
                    sumx=sum_x, sumsq=sum_x_sq, raw=jain, minb=1.0/n)
    
    jain_clipped = jnp.clip(jain, 1.0 / n, 1.0)
    
    # Debug: print final value
    jax.debug.print("JAIN RESULT: raw={raw}, clipped={clipped}", raw=jain, clipped=jain_clipped)
    
    return jain_clipped


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
    
    # Debug: print input returns
    jax.debug.print("=" * 50)
    jax.debug.print("COMPUTE_FAIRNESS_METRICS INPUT: returns={ret}", ret=returns)
    jax.debug.print("  agent returns: {ret}", ret=returns)
    jax.debug.print("  min return: {min}, max return: {max}, mean return: {mean}", 
                    min=jnp.min(returns), max=jnp.max(returns), mean=jnp.mean(returns))
    
    # Mean return (α=0)
    mean_return = jnp.mean(returns)
    
    # Geometric Mean / Nash Social Welfare (α=1)
    # Add small epsilon to avoid log(0)
    eps = 1e-6
    geomean_return = jnp.exp(jnp.mean(jnp.log(jnp.maximum(returns, eps))))
    
    # Minimum return (α→∞)
    min_return = jnp.min(returns)
    
    # Maximum return
    max_return = jnp.max(returns)
    
    # Debug: print basic stats
    jax.debug.print("BASIC STATS: mean={mean}, geomean={geo}, min={min}, max={max}", 
                    mean=mean_return, geo=geomean_return, min=min_return, max=max_return)
    
    gini = gini_coefficient(returns)
    jain = jain_index(returns)
    
    jax.debug.print("=" * 50)
    
    return {
        "mean_return": mean_return,
        "geomean_return": geomean_return,
        "min_return": min_return,
        "max_return": max_return,
        "gini_coefficient": gini,
        "jain_index": jain,
    }

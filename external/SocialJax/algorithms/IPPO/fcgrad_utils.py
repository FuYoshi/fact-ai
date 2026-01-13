"""
FCGrad: Fair Conflict-aware Gradient Adjustment

Based on: "Fair Cooperation in Mixed-Motive Games via Conflict-Aware Gradient Adjustment"
by Woojun Kim and Katia Sycara (Carnegie Mellon University)

This module implements the core FCGrad algorithm that dynamically balances
gradients from individual and collective objectives when they conflict.

Paper: Uses separate networks per agent, compares returns (higher = better).
Lower return = disadvantaged = prioritized.
"""
import jax
import jax.numpy as jnp
from typing import Tuple, Any

PyTree = Any


def flatten_grads(grads: PyTree) -> jnp.ndarray:
    """Flatten a pytree of gradients into a single 1D array."""
    leaves, _ = jax.tree_util.tree_flatten(grads)
    return jnp.concatenate([leaf.ravel() for leaf in leaves])


def unflatten_grads(flat_grads: jnp.ndarray, template: PyTree) -> PyTree:
    """Unflatten a 1D array back into a pytree matching the template structure."""
    leaves, treedef = jax.tree_util.tree_flatten(template)
    
    shapes = [leaf.shape for leaf in leaves]
    sizes = [leaf.size for leaf in leaves]
    
    split_grads = jnp.split(flat_grads, jnp.cumsum(jnp.array(sizes[:-1])))
    reshaped_grads = [g.reshape(s) for g, s in zip(split_grads, shapes)]
    
    return jax.tree_util.tree_unflatten(treedef, reshaped_grads)


def project_gradient(g: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    """
    Project gradient g onto the normal plane of n.
    
    Formula: g_proj = g - (g · n / ||n||²) * n
    
    This removes the component of g that conflicts with n.
    """
    n_norm_sq = jnp.dot(n, n) + 1e-8  # Avoid division by zero
    projection = (jnp.dot(g, n) / n_norm_sq) * n
    return g - projection


def fcgrad_adjust(
    grad_individual: PyTree,
    grad_collective: PyTree,
    value_individual: float,
    value_collective: float,
) -> PyTree:
    """
    Apply FCGrad adjustment to gradients.
    
    Algorithm (from paper):
    1. Check for conflict: dot(grad_ind, grad_col) < 0
    2. If no conflict: use individual gradient (standard RL)
    3. If conflict, prioritize the LOWER objective (disadvantaged):
       - If individual is disadvantaged (lower return): 
         Project INDIVIDUAL onto COLLECTIVE's normal plane
       - If collective is disadvantaged (lower return):
         Project COLLECTIVE onto INDIVIDUAL's normal plane
       - Use ONLY the projected gradient as final update
    
    Paper quote: "FCGrad prioritizes the gradient associated with the lower 
    objective value. For example, if the individual objective is lower than 
    the collective objective, indicating that the agent is in an unfair 
    situation, we project the individual gradient onto the normal plane of 
    the collective gradient."
    
    Args:
        grad_individual: Gradient from individual objective
        grad_collective: Gradient from collective objective  
        value_individual: Current individual return/value (higher = better)
        value_collective: Current collective return/value (higher = better)
    
    Returns:
        Adjusted gradient pytree
    """
    # Flatten gradients for vector operations
    g_ind = flatten_grads(grad_individual)
    g_col = flatten_grads(grad_collective)
    
    # Detect conflict
    dot_product = jnp.dot(g_ind, g_col)
    has_conflict = dot_product < 0
    
    # Determine which objective is disadvantaged (lower return = worse)
    # Paper: "if the individual objective is lower... the agent is in an unfair situation"
    individual_disadvantaged = value_individual < value_collective
    
    def no_conflict_case():
        # No conflict: use individual gradient (standard RL behavior)
        # Could also use g_ind + g_col, but paper implies individual-focused
        return g_ind
    
    def conflict_individual_disadvantaged():
        # Individual has lower return, prioritize individual
        # Project INDIVIDUAL onto COLLECTIVE's normal plane
        g_ind_proj = project_gradient(g_ind, g_col)
        return g_ind_proj
    
    def conflict_collective_disadvantaged():
        # Collective has lower return, prioritize collective  
        # Project COLLECTIVE onto INDIVIDUAL's normal plane
        g_col_proj = project_gradient(g_col, g_ind)
        return g_col_proj
    
    # Apply FCGrad logic
    adjusted_flat = jax.lax.cond(
        has_conflict,
        lambda: jax.lax.cond(
            individual_disadvantaged,
            conflict_individual_disadvantaged,
            conflict_collective_disadvantaged,
        ),
        no_conflict_case,
    )
    
    # Unflatten back to pytree
    return unflatten_grads(adjusted_flat, grad_individual)


def fcgrad_adjust_simple(
    grad_individual: PyTree,
    grad_collective: PyTree,
) -> PyTree:
    """
    Simplified FCGrad without fairness-based prioritization.
    
    Always prioritizes individual objective when there's conflict.
    Projects individual gradient onto collective's normal plane,
    ensuring individual improvement doesn't hurt collective.
    
    Use this version when you don't have access to objective values.
    """
    g_ind = flatten_grads(grad_individual)
    g_col = flatten_grads(grad_collective)
    
    dot_product = jnp.dot(g_ind, g_col)
    has_conflict = dot_product < 0
    
    def no_conflict_case():
        # No conflict: just use individual gradient
        return g_ind
    
    def conflict_case():
        # Conflict: project individual onto collective's normal plane
        g_ind_proj = project_gradient(g_ind, g_col)
        return g_ind_proj
    
    adjusted_flat = jax.lax.cond(has_conflict, conflict_case, no_conflict_case)
    
    return unflatten_grads(adjusted_flat, grad_individual)

"""
FCGrad: Fair Conflict-aware Gradient Adjustment

Based on: "Fair Cooperation in Mixed-Motive Games via Conflict-Aware Gradient Adjustment"
by Woojun Kim and Katia Sycara (Carnegie Mellon University)

This module implements the core FCGrad algorithm that dynamically balances
gradients from individual and collective objectives when they conflict.

Paper: Uses separate networks per agent, compares returns (higher = better).
Lower return = disadvantaged = prioritized.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax import flatten_util

PyTree = Any


def beta_weighting(g_ind: PyTree, g_col: PyTree, beta: float = 0.5) -> PyTree:
    """Compute the beta-weighted sum of two pytrees.

    Args:
        g_ind (PyTree): gradient of individual objective.
        g_col (PyTree): gradient of collective objective.

    Returns:
        (PyTree): beta weighted components.
    """
    assert 0 <= beta <= 1, "beta must be in [0, 1]"
    return jax.tree_map(lambda gi, gc: beta * gi + (1 - beta) * gc, g_ind, g_col)


def pytree_dot(g_ind: PyTree, g_col: PyTree) -> jnp.ndarray:
    """Compute the dot product of two pytrees.

    Args:
        g_ind (PyTree): pytree for individual objective.
        g_col (PyTree): pytree for collective objective.

    Returns:
        (jnp.ndarray): dot product (jax represents it as 0D array).
    """
    g_ind_vec, _ = flatten_util.ravel_pytree(g_ind)
    g_col_vec, _ = flatten_util.ravel_pytree(g_col)
    return jnp.dot(g_ind_vec, g_col_vec)


def pytree_project(g_ind: PyTree, g_col: PyTree, eps: float = 1e-8) -> PyTree:
    """Project g_ind onto the normal plane of g_col.

    Args:
        g_ind (PyTree): gradient of individual objective.
        g_col (PyTree): gradient of collective objective.
        eps (float, optional): avoid division by zero. Defaults to 1e-8.

    Returns:
        (PyTree): g_ind projected onto g_col.
    """
    # Flatten PyTrees to vectors
    g_ind_vec, unravel_fn = flatten_util.ravel_pytree(g_ind)
    g_col_vec, _ = flatten_util.ravel_pytree(g_col)

    dot = jnp.dot(g_ind_vec, g_col_vec)

    projection = g_ind_vec - (dot / (jnp.dot(g_col_vec, g_col_vec) + eps)) * g_col_vec

    # Convert back to PyTree
    g_ind_proj = unravel_fn(projection)
    return g_ind_proj


def grads_align(dot: jnp.ndarray) -> jnp.ndarray:
    """Check if gradients align (dot product >= 0)."""
    return dot >= 0


def collective_disadvantaged(val_ind: jnp.ndarray, val_col: jnp.ndarray) -> jnp.ndarray:
    """Check if collective objective is disadvantaged (lower return).

    Args:
        val_ind (jnp.ndarray): values of individual objective per agent per environment.
        val_col (jnp.ndarray): values of collective objective per agent per environment.

    Returns:
        (jnp.ndarray): array of booleans indicating where collective is at disadvantage.
    """
    return jnp.less(val_col, val_ind)


def fcgrad_adjust(
    g_ind: PyTree,
    g_col: PyTree,
    val_ind: jnp.ndarray,
    val_col: jnp.ndarray,
    beta: float = 0.5,
) -> PyTree:
    """
    Apply FCGrad adjustment to gradients.

    Algorithm (from paper):
    1. Check for conflict: dot(g_ind, g_col) < 0
    2. If no conflict: use beta weighting (see figure 1 of paper).
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
        g_ind: policy gradient from individual objective
        g_col: policy gradient from collective objective
        val_ind (jnp.ndarray): individual objective value as 0D jax array (scalar).
        val_col (jnp.ndarray): collective objective value as 0D jax array (scalar).
        beta (float): beta used for beta weighting on aligned gradients.

    Returns:
        (PyTree): pytree with adjusted gradients.
    """

    def aligned(_):
        # TODO: if g_col is zero, should we just return g_ind?
        return beta_weighting(g_ind, g_col, beta)

    def conflict(_):
        return jax.lax.cond(
            collective_disadvantaged(val_ind, val_col),
            lambda _: pytree_project(g_col, g_ind),
            lambda _: pytree_project(g_ind, g_col),
            operand=None,
        )

    dot = pytree_dot(g_ind, g_col)
    grad_fcgrad = jax.lax.cond(grads_align(dot), aligned, conflict, operand=None)
    return grad_fcgrad


def policy_loss(
    params: PyTree,
    traj_batch: PyTree,
    advantages: jnp.ndarray,
    clip_eps: float,
    network_used,
) -> jnp.ndarray:
    """Compute the policy loss based on the advantages.

    Args:
        params (PyTree): pytree of network parameters.
        traj_batch (Transition): pytree of trajectory data.
        advantages (jnp.ndarray): advantages of the actor.
        clip_eps (float): epsilon parameter for PPO clipping.
        network_used: network used to compute policy loss from.

    Returns:
        (jnp.ndarray): scalar loss represented as 0D jax array.
    """
    pi, _, _ = network_used.apply(params, traj_batch.obs)
    log_prob = pi.log_prob(traj_batch.action)
    ratio = jnp.exp(log_prob - traj_batch.log_prob)

    # clipped surrogate objectives
    unclipped = ratio * advantages
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -jnp.mean(jnp.minimum(unclipped, clipped))


def compute_fcgrad(
    params: PyTree,
    traj_batch: PyTree,
    adv_ind: jnp.ndarray,
    adv_col: jnp.ndarray,
    targets_ind: jnp.ndarray,
    targets_col: jnp.ndarray,
    clip_eps: float,
    beta: float,
    network_used,
) -> PyTree:
    """Apply FCGrad by computing the policy gradients g_ind, g_col and then using
    the FCGrad algorithm. Each actor uses their own V_ind, V_col in the FCGrad
    algorithm. These per-actor gradients are then aggregated by taking the mean.

    Args: (B = batch_size)
        params (PyTree): pytree of network parameters.
        traj_batch (Transition): pytree of trajectory data.
        adv_ind (jnp.ndarray): individual advantages. [B]
        adv_col (jnp.ndarray): collective advantages. [B]
        targets_ind (jnp.ndarray): individual targets. [B]
        targets_col (jnp.ndarray): collective targets. [B]
        clip_eps (float): epsilon parameter for PPO clipping.
        beta (float): beta parameter for beta weighting in FCGrad.
        network_used: network used to compute policy loss from.

    Returns:
        (PyTree): pytree with adjusted gradients.
    """

    # Compute the gradient using the adavantages.
    grad_fn = jax.value_and_grad(policy_loss)

    # Compute policy gradients over batch (already averaged in policy_loss)
    loss_ind, g_ind = grad_fn(params, traj_batch, adv_ind, clip_eps, network_used)
    loss_col, g_col = grad_fn(params, traj_batch, adv_col, clip_eps, network_used)

    # TODO:
    val_ind = ...
    val_col = ...

    g_policy = fcgrad_adjust(g_ind, g_col, val_ind, val_col, beta)

    # Maybe return policy loss for logging?
    return g_policy


# ===================================================
# Unit Tests
# ===================================================


def test_projection():
    """Test that projection onto normal plane works correctly."""
    # Parallel case: ignored (no projection if gradients align)

    # Perpendicular case: projection = g
    g = jnp.array([1.0, 0.0])
    n = jnp.array([0.0, 1.0])
    proj = pytree_project(g, n)
    assert jnp.allclose(proj, g), f"Perpendicular case failed: {proj}"

    # General case: projection removes component along n
    g = jnp.array([1.0, 1.0])
    n = jnp.array([1.0, 0.0])
    proj = pytree_project(g, n)
    assert jnp.allclose(
        proj, jnp.array([0.0, 1.0]), atol=1e-6
    ), f"General case failed: {proj}"

    # Collective zero gradient: should return individual unchanged
    g = {"w": jnp.array([1.0, 2.0, 3.0])}
    n = {"w": jnp.array([0.0, 0.0, 0.0])}  # zero gradient
    proj = pytree_project(g, n)
    assert jnp.allclose(
        proj["w"], g["w"]
    ), f"Zero n case failed: {proj['w']}, expected {g['w']}"

    print("Projection tests passed")


def test_conflict_detection():
    """Test that conflicts are detected correctly."""
    # Conflicting gradients (dot < 0)
    g_ind = {"w": jnp.array([1.0, 0.0])}
    g_col = {"w": jnp.array([-1.0, 0.0])}

    dot = pytree_dot(g_ind, g_col)
    assert not grads_align(dot), f"Conflicting case failed, got dot={dot}"

    # Non-conflicting gradients (dot >= 0)
    g_ind = {"w": jnp.array([1.0, 0.0])}
    g_col = {"w": jnp.array([1.0, 1.0])}

    dot = pytree_dot(g_ind, g_col)
    assert grads_align(dot), f"Non-conflicting case failed, got dot={dot}"

    print("Conflict detection tests passed")


def test_prioritization():
    """Test that the correct objective is prioritized based on return value."""
    # Create conflicting gradients
    g_ind = {"w": jnp.array([1.0, 0.0])}
    g_col = {"w": jnp.array([-0.5, 0.5])}

    # Case 1: Individual value < collective value (individual disadvantaged)
    # Should project INDIVIDUAL onto COLLECTIVE's normal plane
    result = fcgrad_adjust(g_ind, g_col, val_ind=0.5, val_col=1.0)
    dot_with_col = pytree_dot(result, g_col)

    assert (
        dot_with_col >= -1e-6
    ), f"Projected grad should not conflict with collective: {dot_with_col}"

    # Case 2: Collective value < individual value (collective disadvantaged)
    # Should project COLLECTIVE onto INDIVIDUAL's normal plane
    result = fcgrad_adjust(g_ind, g_col, val_ind=1.0, val_col=0.5)
    dot_with_ind = pytree_dot(result, g_ind)

    assert (
        dot_with_ind >= -1e-6
    ), f"Projected grad should not conflict with individual: {dot_with_ind}"

    print("Prioritization tests passed")


def test_nested_pytree():
    """Test with nested parameter structure like real neural networks."""
    g_ind = {
        "params": {
            "Dense_0": {
                "kernel": jnp.array([[1.0, -1.0], [0.5, 0.5]]),
                "bias": jnp.array([0.1, -0.1]),
            },
            "Dense_1": {"kernel": jnp.array([[0.2]]), "bias": jnp.array([0.3])},
        }
    }
    g_col = {
        "params": {
            "Dense_0": {
                "kernel": jnp.array([[-0.5, 0.5], [-0.2, 0.2]]),
                "bias": jnp.array([-0.05, 0.05]),
            },
            "Dense_1": {"kernel": jnp.array([[0.1]]), "bias": jnp.array([0.15])},
        }
    }

    # Just check it doesn't crash and returns correct structure
    result = fcgrad_adjust(g_ind, g_col, val_ind=1.0, val_col=2.0)

    assert "params" in result
    assert "Dense_0" in result["params"]
    assert "kernel" in result["params"]["Dense_0"]
    assert result["params"]["Dense_0"]["kernel"].shape == (2, 2)

    print("Nested pytree test passed")


def test_grad_col_zero():
    """Test with nested parameter structure like real neural networks."""
    beta = 0.5
    g_ind = {"w": jnp.array([1.0, 2.0, 3.0])}
    g_col = {"w": jnp.array([0.0, 0.0, 0.0])}
    # Just check it doesn't crash and returns correct structure
    result = fcgrad_adjust(g_ind, g_col, val_ind=1.0, val_col=2.0, beta=beta)

    assert jnp.allclose(
        result["w"], (1 - beta) * g_ind["w"]
    ), f"Expected (1 - beta) * g_ind['w'], got {result['w']}"


if __name__ == "__main__":
    print("=" * 50)
    print("FCGrad Unit Tests")
    print("=" * 50 + "\n")

    test_projection()
    test_conflict_detection()
    test_prioritization()
    test_nested_pytree()
    test_grad_col_zero()

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)

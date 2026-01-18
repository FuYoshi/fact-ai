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
        val_ind (jnp.ndarray): values of individual objective.
        val_col (jnp.ndarray): values of collective objective.

    Returns:
        (jnp.ndarray): True if collective is disadvantaged, False otherwise.
    """
    return val_col < val_ind


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


def combined_losses(
    params: PyTree,
    traj_batch: PyTree,
    advantages_ind: jnp.ndarray,
    advantages_col: jnp.ndarray,
    targets_ind: jnp.ndarray,
    targets_col: jnp.ndarray,
    clip_eps: float,
    network_used,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[float, PyTree, PyTree],
]:
    """Do a forward pass through network_used and compute the actor and critic
    losses. Also returns auxilary data containing the total_loss and the values
    of the individual and collective objective for each sample.

    Args:
        params (PyTree): parameters of the ActorCritic network.
        traj_batch (Transition): batch of trajectories.
        advantages_ind (jnp.ndarray): batch of individual advantages.
        advantages_col (jnp.ndarray): batch of collective advantages.
        targets_ind (jnp.ndarray): batch of individual targets.
        targets_col (jnp.ndarray): batch of collective targets.
        clip_eps (float): epsilon parameter for PPO clipping.
        network_used: ActorCritic network for the forward pass.

    Returns:
        A tuple containing:
            - individual/collective losses for the actor/critic.
            - auxilary data consisting of:
                - total_loss
                - individual objective values of the batch.
                - collective objective values of the batch.
    """

    def _policy_loss(
        pi,
        traj_batch: PyTree,
        advantages: jnp.ndarray,
        clip_eps: float,
    ) -> jnp.ndarray:
        """Compute the policy loss based on the advantages.

        Args:
            policy (distrax.Categorical): policy distribution.
            traj_batch (Transition): pytree of trajectory data.
            advantages (jnp.ndarray): advantages of the actor.
            clip_eps (float): epsilon parameter for PPO clipping.

        Returns:
            (jnp.ndarray): scalar loss represented as 0D jax array.
        """
        log_prob = pi.log_prob(traj_batch.action)
        ratio = jnp.exp(log_prob - traj_batch.log_prob)

        # clipped surrogate objectives
        unclipped = ratio * advantages
        clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        return -jnp.mean(jnp.minimum(unclipped, clipped))

    def _value_loss(baseline, value, targets, clip_eps):
        """Compute the value loss based on the targets.

        Args:
            baseline (jnp.ndarray): value predictions from the behavior (old) policy used during rollout.
            value (jnp.ndarray): value predictions from the current critic network.
            targets (jnp.ndarray): target value estimates (e.g. returns or GAE targets).
            clip_eps (float): epsilon parameter for PPO clipping.

        Returns:
            (jnp.ndarray): scalar loss represented as 0D jax array.
        """
        value_pred_clipped = baseline + (value - baseline).clip(-clip_eps, clip_eps)
        val_losses_unclipped = jnp.square(value - targets)
        val_losses_clipped = jnp.square(value_pred_clipped - targets)
        val_loss = 0.5 * jnp.maximum(val_losses_unclipped, val_losses_clipped).mean()
        return val_loss

    pi, val_ind, val_col = network_used.apply(params, traj_batch.obs)
    loss_actor_ind = _policy_loss(pi, traj_batch, advantages_ind, clip_eps)
    loss_actor_col = _policy_loss(pi, traj_batch, advantages_col, clip_eps)
    loss_critic_ind = _value_loss(traj_batch.value, val_ind, targets_ind, clip_eps)
    loss_critic_col = _value_loss(traj_batch.value, val_col, targets_col, clip_eps)

    losses = loss_actor_ind, loss_actor_col, loss_critic_ind, loss_critic_col
    aux = (sum(losses), val_ind, val_col)
    return (losses, aux)


def compute_fcgrad_jacobian(
    params: PyTree,
    traj_batch: PyTree,
    adv_ind: jnp.ndarray,
    adv_col: jnp.ndarray,
    targets_ind: jnp.ndarray,
    targets_col: jnp.ndarray,
    clip_eps: float,
    beta: float,
    network_used,
) -> tuple[PyTree, float]:
    """Compute the gradients and total loss update. Actor gradients are
    computed by using advantage-based policy gradient estimation. Critic
    gradients are computed using value function regression based on targets.
    Losses make use of PPO clipping for stable updates.

    Note:
        This function prevents going through the forward pass multiple times
        by computing the jacobian (all losses w.r.t. params) instead. Since the
        losses are scalars, the shape of the jacobian is [4 x #params].

        This function might still take longer to compute than compute_fcgrad(),
        because compiling will take longer. Also, performing multiple forward
        passes in each minibatch is probably not as expensive as I think.

    Args:
        params (PyTree): parameters of the ActorCritic network.
        traj_batch (Transition): batch of trajectories.
        adv_ind (jnp.ndarray): batch of individual advantages.
        adv_col (jnp.ndarray): batch of collective advantages.
        targets_ind (jnp.ndarray): batch of individual targets.
        targets_col (jnp.ndarray): batch of collective targets.
        clip_eps (float): epsilon parameter for PPO clipping.
        beta (float): beta parameter for FCGrad beta weighting.
        network_used: ActorCritic network for the forward pass.

    Returns:
        A tuple containing:
            - grads (for actor/critic for individual/collective objective).
            - total_loss
    """
    # Compute the actor and critic losses.
    loss_fn = jax.jacrev(combined_losses, has_aux=True)
    jacobian, aux = loss_fn(
        params,
        traj_batch,
        adv_ind,
        adv_col,
        targets_ind,
        targets_col,
        clip_eps,
        network_used,
    )
    g_actor_ind, g_actor_col, g_critic_ind, g_critic_col = jacobian
    total_loss, val_ind, val_col = aux

    # Apply FCGrad to each sample in the minibatch.
    grads_fn = jax.vmap(fcgrad_adjust, in_axes=(None, None, 0, 0, None))
    g_actor_per_sample = grads_fn(g_actor_ind, g_actor_col, val_ind, val_col, beta)

    # Aggregate actor gradients by taking the mean.
    g_actor = jax.tree_map(lambda x: jnp.mean(jnp.stack(x), axis=0), g_actor_per_sample)

    grads = jax.tree_map(lambda a, b, c: a + b + c, g_actor, g_critic_ind, g_critic_col)
    return grads, total_loss


def policy_loss(
    params: PyTree,
    traj_batch: PyTree,
    advantages: jnp.ndarray,
    clip_eps: float,
    network_used,
    individual: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the policy loss based on the advantages.

    Args:
        params (PyTree): pytree of network parameters.
        traj_batch (Transition): pytree of trajectory data.
        advantages (jnp.ndarray): advantages of the actor.
        clip_eps (float): epsilon parameter for PPO clipping.
        network_used: ActorCritic network for forward pass.
        individual (bool): bool to determine which value to return as aux.

    Returns:
        (jnp.ndarray): scalar loss represented as 0D jax array.
    """
    pi, val_ind, val_col = network_used.apply(params, traj_batch.obs)
    log_prob = pi.log_prob(traj_batch.action)
    ratio = jnp.exp(log_prob - traj_batch.log_prob)

    # clipped surrogate objectives
    unclipped = ratio * advantages
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages

    aux = val_ind if individual else val_col
    return -jnp.mean(jnp.minimum(unclipped, clipped)), aux


def value_loss(
    params: PyTree,
    traj_batch: PyTree,
    targets: jnp.ndarray,
    clip_eps: float,
    network_used,
    individual: bool = True,
) -> jnp.ndarray:
    """Compute the value loss based on the targets.

    Args:
        params (PyTree): pytree of network parameters.
        traj_batch (Transition): pytree of trajectory data.
        targets (jnp.ndarray): target value estimates (e.g. returns or GAE targets).
        clip_eps (float): epsilon parameter for PPO clipping.
        network_used: ActorCritic network for forward pass.
        individual (bool): bool to determine which value loss to compute.

    Returns:
        A tuple containing:
            (jnp.ndarray): scalar loss represented as 0D jax array.
            (jnp.ndarray): critic network values for this batch.
    """
    _, val_ind, val_col = network_used.apply(params, traj_batch.obs)

    value = val_ind if individual else val_col

    baseline = traj_batch.value
    value_pred_clipped = baseline + (value - baseline).clip(-clip_eps, clip_eps)
    val_losses_unclipped = jnp.square(value - targets)
    val_losses_clipped = jnp.square(value_pred_clipped - targets)
    val_loss = 0.5 * jnp.maximum(val_losses_unclipped, val_losses_clipped).mean()
    return val_loss


def compute_fcgrad(
    params: PyTree,
    traj_batch: PyTree,
    adv_ind: jnp.ndarray,
    adv_col: jnp.ndarray,
    targets_ind: jnp.ndarray,
    targets_col: jnp.ndarray,
    clip_eps: float,
    beta: float,
    network,
) -> tuple[PyTree, float]:
    """Compute the gradients and total loss update. Actor gradients are
    computed by using advantage-based policy gradient estimation. Critic
    gradients are computed using value function regression based on targets.
    Losses make use of PPO clipping for stable updates.

    Args:
        params (PyTree): parameters of the ActorCritic network.
        traj_batch (Transition): batch of trajectories.
        adv_ind (jnp.ndarray): batch of individual advantages.
        adv_col (jnp.ndarray): batch of collective advantages.
        targets_ind (jnp.ndarray): batch of individual targets.
        targets_col (jnp.ndarray): batch of collective targets.
        clip_eps (float): epsilon parameter for PPO clipping.
        beta (float): beta parameter for FCGrad beta weighting.
        network_used: ActorCritic network for the forward pass.

    Returns:
        A tuple containing:
            - grads (for actor/critic for individual/collective objective).
            - total_loss
    """
    # Actor gradients/loss.
    actor_grad_fn = jax.value_and_grad(policy_loss, has_aux=True)
    (loss_actor_ind, v_ind), g_ind = actor_grad_fn(params, traj_batch, adv_ind, clip_eps, network, True)
    (loss_actor_col, v_col), g_col = actor_grad_fn(params, traj_batch, adv_col, clip_eps, network, False)

    # Vectorize FCGrad adjustment for each value sample in batch.
    grads_fn = jax.vmap(fcgrad_adjust, in_axes=(None, None, 0, 0, None))
    g_actor_per_sample = grads_fn(g_ind, g_col, v_ind, v_col, beta)

    # Aggregate actor gradients by taking the mean.
    g_actor = jax.tree_map(lambda x: jnp.mean(jnp.stack(x), axis=0), g_actor_per_sample)

    # Critic gradients/loss
    critic_grad_fn = jax.value_and_grad(value_loss)
    loss_critic_ind, g_critic_ind = critic_grad_fn(params, traj_batch, targets_ind, clip_eps, network, True)
    loss_critic_col, g_critic_col = critic_grad_fn(params, traj_batch, targets_col, clip_eps, network, False)

    # Combining gradients/loss (they should have zeroes for other heads).
    grads = jax.tree_map(lambda a, b, c: a + b + c, g_actor, g_critic_ind, g_critic_col)
    total_loss = loss_actor_ind + loss_actor_col + loss_critic_ind + loss_critic_col
    return grads, total_loss


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

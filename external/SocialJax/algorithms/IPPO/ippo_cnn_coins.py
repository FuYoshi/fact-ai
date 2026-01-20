"""
IPPO Training for Coin Game with Proper Fairness Metrics
Based on PureJaxRL & JaxMARL implementation

Key improvements:
- Proper handling of original_rewards for fairness computation
- Cleaner code structure with helper functions
- Better documentation and type hints
"""
import copy
import os
import pickle
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

# Add SocialJax to path
current_dir = os.path.dirname(os.path.abspath(__file__))
socialjax_path = os.path.join(current_dir, '..', '..')
sys.path.insert(0, os.path.abspath(socialjax_path))

import socialjax
from fairness_metrics import compute_fairness_metrics
# Local imports
from fcgrad_utils import compute_fcgrad
from socialjax.wrappers.baselines import LogWrapper

# =============================================================================
# Network Architecture
# =============================================================================

class CNN(nn.Module):
    """Convolutional feature extractor."""
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        x = nn.Conv(features=32, kernel_size=(5, 5),
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Conv(features=32, kernel_size=(3, 3),
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Conv(features=32, kernel_size=(3, 3),
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        return x


class ActorHead(nn.Module):
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        return distrax.Categorical(logits=x)


class CriticHead(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return jnp.squeeze(x, axis=-1)


class ActorCritic(nn.Module):
    """Actor-Critic network with optional dual critic heads."""
    action_dim: int
    activation: str = "relu"
    embedder: str = "embedder"
    actor_head: str = "actor_head"
    critic_head_ind: str = "critic_head_ind"
    critic_head_col: str = "critic_head_col"
    use_dual_critic: bool = False

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        embedding = CNN(self.activation, name=self.embedder)(x)

        pi = ActorHead(self.action_dim, activation=activation, name=self.actor_head)(embedding)

        critic_ind = CriticHead(activation=activation, name=self.critic_head_ind)(embedding)

        if self.use_dual_critic:
            critic_col = CriticHead(activation=activation, name=self.critic_head_col)(embedding)
            return pi, critic_ind, critic_col
        return pi, critic_ind, critic_ind


# =============================================================================
# Data Structures
# =============================================================================

class Transition(NamedTuple):
    """Single transition in a trajectory."""
    done: jnp.ndarray
    action: jnp.ndarray
    value_ind: jnp.ndarray
    value_col: jnp.ndarray
    reward_ind: jnp.ndarray
    reward_col: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Dict[str, jnp.ndarray]
    # Store original individual rewards separately for fairness computation
    original_rewards: jnp.ndarray


# =============================================================================
# Utility Functions
# =============================================================================

def batchify(x: jnp.ndarray, agent_list, num_actors: int) -> jnp.ndarray:
    """Stack and reshape observations for batched processing."""
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))


def batchify_dict(x: dict, agent_list, num_actors: int) -> jnp.ndarray:
    """Stack dict values and reshape for batched processing."""
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs: int, num_agents: int) -> dict:
    """Reshape batched array back to per-agent dict."""
    x = x.reshape((num_agents, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def process_info_dict(
    info: Dict[str, jnp.ndarray],
    num_actors: int,
    num_envs: int,
    num_agents: int
) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
    """
    Process info dict from environment step.

    Handles original_rewards separately since it has shape (num_envs, num_agents)
    while other info values have shape (num_envs,) per agent.

    Returns:
        info_processed: Dict with values reshaped to (num_actors,)
        original_rewards: Array of shape (num_actors,) in agent-major order
                         [agent0_env0, agent0_env1, ..., agent1_env0, ...]
    """
    info_processed = {}
    original_rewards = None

    for key, value in info.items():
        if key == "original_rewards":
            # original_rewards shape: (num_envs, num_agents)
            # Convert to agent-major order: (num_agents, num_envs) -> (num_actors,)
            original_rewards = value.T.reshape(-1)
        else:
            # Standard info values: reshape to (num_actors,)
            info_processed[key] = value.reshape((num_actors,))

    # If original_rewards not provided, create zeros
    if original_rewards is None:
        original_rewards = jnp.zeros(num_actors)

    return info_processed, original_rewards


def compute_rollout_returns(
    rewards: jnp.ndarray,
    original_rewards: jnp.ndarray,
    num_envs: int,
    num_agents: int,
    use_original: bool
) -> jnp.ndarray:
    """
    Compute per-agent rollout returns for fairness metrics.

    Args:
        rewards: Training rewards, shape (num_steps, num_actors)
        original_rewards: Individual rewards before sharing, shape (num_steps, num_actors)
        num_envs: Number of parallel environments
        num_agents: Number of agents
        use_original: Whether to use original_rewards (True for shared_rewards mode)

    Returns:
        rollout_returns: Per-agent returns, shape (num_agents,)
    """
    # Select which rewards to use for fairness computation
    reward_source = original_rewards if use_original else rewards

    # reward_source shape: (num_steps, num_actors)
    # num_actors = num_agents * num_envs in agent-major order
    # Agent-major: [agent0_env0, agent0_env1, ..., agent0_envN-1, agent1_env0, ...]

    # Reshape to (num_steps, num_agents, num_envs)
    reward_reshaped = reward_source.reshape(reward_source.shape[0], num_agents, num_envs)

    # Sum over steps (axis=0), then mean over envs (axis=-1) -> (num_agents,)
    rollout_returns = reward_reshaped.sum(axis=0).mean(axis=-1)

    return rollout_returns


def compute_discounted_returns(rewards, dones, last_value, gamma=0.99) -> jnp.ndarray:
    """
    Compute returns with discounts (gamma).

    Args:
        rewards: training rewards, shape (num_steps, num_actors).
        dones: bool flags indicating if episode ended (num_steps, num_actors).
        last_val: last value in batch.
        gamma: discount factor.

    Returns:
        returns computed under the initial state distribution.
    """
    num_steps, num_actors = rewards.shape
    returns = jnp.zeros_like(rewards)

    def scan_fn(carry: jnp.ndarray, inputs: tuple):
        reward, done = inputs
        ret = reward + gamma * carry * (1.0 - done)
        return ret, ret

    # start from the last step (bootstrapped)
    _, returns = jax.lax.scan(
        scan_fn,
        last_value,
        (rewards, dones),
        reverse=True
    )
    # TODO: might have to reverse result if you want to plot.
    return returns


# =============================================================================
# GAE Computation
# =============================================================================

def compute_gae(
    traj_batch: Transition,
    last_value: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
    individual: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Generalized Advantage Estimation.

    Args:
        traj_batch: Trajectory of transitions
        last_value: Value estimate for final state
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        individual: use individual or collective value and reward.

    Returns:
        advantages: GAE advantages
        targets: Value targets (advantages + values)
    """
    def _get_advantages(gae_and_next_value, transition: Transition):
        gae, next_value = gae_and_next_value
        done = transition.done
        if individual:
            value, reward = transition.value_ind, transition.reward_ind
        else:
            value, reward = transition.value_col, transition.reward_col

        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae

        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_value), last_value),
        traj_batch,
        reverse=True,
        unroll=16,
    )

    if individual:
        value = traj_batch.value_ind
    else:
        value = traj_batch.value_col

    targets = advantages + value
    return advantages, targets


# =============================================================================
# Loss Functions
# =============================================================================

def ppo_loss(
    params,
    traj_batch: Transition,
    advantages: jnp.ndarray,
    targets: jnp.ndarray,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    network: nn.Module
) -> Tuple[jnp.ndarray, Tuple]:
    """
    Compute PPO loss.

    Returns:
        total_loss: Combined actor, critic, and entropy loss
        aux: Tuple of (value_loss, actor_loss, entropy)
    """
    pi, value, _ = network.apply(params, traj_batch.obs)
    log_prob = pi.log_prob(traj_batch.action)

    # Value loss with clipping
    value_pred_clipped = traj_batch.value_ind + (value - traj_batch.value_ind).clip(-clip_eps, clip_eps)
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    # Actor loss with clipping
    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    loss_actor1 = ratio * advantages_normalized
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_normalized
    actor_loss = -jnp.minimum(loss_actor1, loss_actor2).mean()

    # Entropy bonus
    entropy = pi.entropy().mean()

    total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy

    return total_loss, (value_loss, actor_loss, entropy)


# =============================================================================
# Main Training Function
# =============================================================================

def make_train(config: Dict, pbar: Optional[tqdm] = None):
    """Create the training function."""

    # Environment setup
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = LogWrapper(env, replace_info=False)

    num_agents = env.num_agents
    num_envs = config["NUM_ENVS"]
    num_actors = num_agents * num_envs if config["PARAMETER_SHARING"] else num_envs

    # Compute derived config values
    config["NUM_ACTORS"] = num_actors
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // num_envs
    config["MINIBATCH_SIZE"] = num_actors * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]

    # Check if using shared rewards (for fairness computation)
    use_shared_rewards = config.get("ENV_KWARGS", {}).get("shared_rewards", False)

    def linear_schedule(count):
        """Linear learning rate annealing."""
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng: jnp.ndarray) -> Dict:
        """Main training loop."""

        # Initialize network
        network = ActorCritic(
            action_dim=env.action_space().n,
            activation=config["ACTIVATION"],
            use_dual_critic=config.get("USE_DUAL_HEAD", False)
        )

        rng, init_rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space()[0].shape))
        network_params = network.init(init_rng, init_x)

        # Optimizer
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # Initialize environment
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset)(reset_rngs)

        # =====================================================================
        # Update Step (one iteration of data collection + policy update)
        # =====================================================================

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, update_step, rng = runner_state

            # -----------------------------------------------------------------
            # Collect Trajectories
            # -----------------------------------------------------------------

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state
                rng, action_rng, step_rng = jax.random.split(rng, 3)

                # Prepare observations: (num_envs, num_agents, ...) -> (num_actors, ...)
                obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(
                    -1, *env.observation_space()[0].shape
                )

                # Get actions from policy
                pi, value_ind, value_col = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=action_rng)
                log_prob = pi.log_prob(action)

                # Convert to environment action format
                env_act = unbatchify(action, env.agents, num_envs, num_agents)
                env_act_list = [v for v in env_act.values()]

                # Step environment
                step_rngs = jax.random.split(step_rng, num_envs)
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    step_rngs, env_state, env_act_list
                )

                # Process info and extract original_rewards
                # reward shape: (num_envs, num_agents)
                # info["original_rewards"] shape: (num_envs, num_agents) when shared_rewards=True
                info_processed, original_rewards = process_info_dict(
                    info, num_actors, num_envs, num_agents
                )

                # Reshape rewards to (num_actors,) in agent-major order
                reward_batch = batchify(reward, env.agents, num_actors).squeeze()
                done_batch = batchify_dict(done, env.agents, num_actors).squeeze()

                # Collective reward is the mean across agents for each env.
                # Tile to match agent-major order: [a0_e0, a0_e1, ..., a1_e0, a1_e1, ...]
                # reward shape: (num_envs, num_agents)
                reward_col_env = jnp.mean(reward, axis=1)  # (num_envs,)
                reward_col = jnp.tile(reward_col_env, num_agents)  # (num_actors,)

                transition = Transition(
                    done=done_batch,
                    action=action,
                    value_ind=value_ind,
                    value_col=value_col,
                    reward_ind=reward_batch,
                    reward_col=reward_col,
                    log_prob=log_prob,
                    obs=obs_batch,
                    info=info_processed,
                    original_rewards=original_rewards,
                )

                runner_state = (train_state, env_state, obsv, update_step, rng)
                return runner_state, transition

            # Collect trajectory
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            train_state, env_state, last_obs, update_step, rng = runner_state

            # -----------------------------------------------------------------
            # Compute Advantages
            # -----------------------------------------------------------------

            last_obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(
                -1, *env.observation_space()[0].shape
            )
            _, last_val_ind, last_val_col = network.apply(train_state.params, last_obs_batch)

            advantages_ind, targets_ind = compute_gae(
                traj_batch, last_val_ind, config["GAMMA"], config["GAE_LAMBDA"]
            )

            # Collective advantages/targets (for FCGrad).
            advantages_col, targets_col = compute_gae(
                traj_batch, last_val_col, config["GAMMA"], config["GAE_LAMBDA"], individual=False
            )

            # -----------------------------------------------------------------
            # Compute Returns
            # -----------------------------------------------------------------

            # Collect trajectories and compute backwards from last value.
            # Shape: (num_envs, num_agents)
            dis_returns_ind = compute_discounted_returns(
                rewards=traj_batch.reward_ind,
                dones=traj_batch.done,
                last_value=last_val_ind,
                gamma=config["GAMMA"],
            )

            dis_returns_col = compute_discounted_returns(
                rewards=traj_batch.reward_col,
                dones=traj_batch.done,
                last_value=last_val_col,
                gamma=config["GAMMA"],
            )

            # -----------------------------------------------------------------
            # Policy Update
            # -----------------------------------------------------------------

            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, adv_ind, adv_col, tgt_ind, tgt_col = batch_info

                    if config.get("FCGRAD", False):
                        grads, loss_info = compute_fcgrad(
                            train_state.params,
                            traj_batch,
                            jnp.mean(dis_returns_ind), jnp.mean(dis_returns_col),
                            adv_ind, adv_col,
                            tgt_ind, tgt_col,
                            config["CLIP_EPS"],
                            config["FCGRAD_BETA"],
                            network
                        )
                    else:
                        grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
                        (loss_ppo, aux), grads = grad_fn(
                            train_state.params,
                            traj_batch,
                            adv_ind, tgt_ind,
                            config["CLIP_EPS"],
                            config["VF_COEF"],
                            config["ENT_COEF"],
                            network
                        )
                        loss_info = {"loss_ppo": loss_ppo}

                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, loss_info

                train_state, traj_batch, adv_ind, adv_col, tgt_ind, tgt_col, rng = update_state
                rng, perm_rng = jax.random.split(rng)

                batch_size = config["NUM_STEPS"] * num_actors
                permutation = jax.random.permutation(perm_rng, batch_size)

                # Prepare batch
                batch = (traj_batch, adv_ind, adv_col, tgt_ind, tgt_col)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )

                train_state, loss_info = jax.lax.scan(
                    _update_minibatch, train_state, minibatches
                )

                update_state = (train_state, traj_batch, adv_ind, adv_col, tgt_ind, tgt_col, rng)
                return update_state, loss_info

            update_state = (train_state, traj_batch, advantages_ind, advantages_col,
                           targets_ind, targets_col, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # -----------------------------------------------------------------
            # Compute Metrics
            # -----------------------------------------------------------------

            # Compute per-agent rollout returns for fairness metrics
            # Use original_rewards when shared_rewards is enabled
            rollout_returns = compute_rollout_returns(
                rewards=traj_batch.reward_ind,
                original_rewards=traj_batch.original_rewards,
                num_envs=num_envs,
                num_agents=num_agents,
                use_original=use_shared_rewards
            )

            # TODO: do we have to track collective fairness metrics?
            # Compute fairness metrics
            fairness = compute_fairness_metrics(rollout_returns)

            # Average other metrics
            metric = jax.tree_map(lambda x: x.mean(), traj_batch.info)

            # Aggregate over loss info for loss metrics.
            loss_metrics = jax.tree_map(lambda x: x.mean(), loss_info)

            # Add fairness metrics and loss metrics.
            metric.update(loss_metrics)
            metric.update(fairness)
            for i in range(num_agents):
                metric[f"agent_{i}_return"] = rollout_returns[i]

            metric["update_step"] = update_step + 1
            metric["env_step"] = (update_step + 1) * config["NUM_STEPS"] * num_envs

            # Scale eat_own_coins by episode length
            if "eat_own_coins" in metric:
                metric["eat_own_coins"] = metric["eat_own_coins"] * config["ENV_KWARGS"]["num_inner_steps"]

            # Logging callback
            def callback(metric):
                def to_native(x):
                    if isinstance(x, (jnp.ndarray, jnp.generic)):
                        return float(x.item()) if x.ndim == 0 else float(x.mean())
                    return x

                metric_converted = jax.tree_map(to_native, metric)
                wandb.log(metric_converted)

                if pbar is not None:
                    pbar.update(1)
                    ret_val = to_native(metric.get('returned_episode_returns', 0))
                    pbar.set_postfix({
                        'return': f"{ret_val:.2f}",
                        'step': int(to_native(metric.get('env_step', 0)))
                    })

            jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, update_step + 1, rng)
            return runner_state, metric

        # Run training loop
        rng, train_rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, train_rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


# =============================================================================
# Evaluation and Checkpointing
# =============================================================================

def save_params(train_state: TrainState, save_path: str):
    """Save model parameters to file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)
    with open(save_path, 'wb') as f:
        pickle.dump(params, f)


def load_params(load_path: str) -> Dict:
    """Load model parameters from file."""
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)


def evaluate(params, env, config: Dict, save_dir: str = "evaluation/coins"):
    """Run evaluation and create GIF."""
    rng = jax.random.PRNGKey(0)
    rng, reset_rng = jax.random.split(rng)

    obs, state = env.reset(reset_rng)

    network = ActorCritic(
        action_dim=env.action_space().n,
        activation=config["ACTIVATION"],
        use_dual_critic=config.get("USE_DUAL_HEAD", False)
    )

    pics = [env.render(state)]

    for t in range(config["GIF_NUM_FRAMES"]):
        rng, action_rng, step_rng = jax.random.split(rng, 3)

        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(
            -1, *env.observation_space()[0].shape
        )

        pi, _, _ = network.apply(params, obs_batch)
        actions = pi.sample(seed=action_rng)

        env_act = {
            k: v.squeeze()
            for k, v in unbatchify(actions, env.agents, 1, env.num_agents).items()
        }

        obs, state, reward, done, info = env.step(
            step_rng, state, [v.item() for v in env_act.values()]
        )

        pics.append(env.render(state))

        print(f"Step {t}: Actions={env_act}, Rewards={reward}")

    # Save GIF
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    pics_pil = [Image.fromarray(np.array(img)) for img in pics]
    gif_path = f"{save_dir}/{env.num_agents}-agents_seed-{config['SEED']}.gif"

    pics_pil[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=pics_pil[1:],
        duration=200,
        loop=0,
    )

    wandb.log({"Episode GIF": wandb.Video(gif_path, caption="Evaluation", format="gif")})
    print(f"Saved GIF to {gif_path}")


# =============================================================================
# Main Entry Points
# =============================================================================

def single_run(config):
    """Run a single training experiment."""
    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'ippo_cnn_coins_seed{config["SEED"]}'
    )

    num_updates = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    print(f"\nStarting training: {num_updates} updates, {config['TOTAL_TIMESTEPS']:,} timesteps")

    pbar = tqdm(total=num_updates, desc="Training", unit="update", mininterval=30)

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    train_fn = jax.jit(make_train(config, pbar=pbar))
    out = jax.vmap(train_fn)(rngs)

    pbar.close()

    # Save and evaluate
    train_state = jax.tree_map(lambda x: x[0], out["runner_state"][0])
    save_path = f"./checkpoints/{config['ENV_NAME']}_seed{config['SEED']}.pkl"
    save_params(train_state, save_path)

    params = load_params(save_path)
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    evaluate(params, env, config)

    wandb.finish()


@hydra.main(version_base=None, config_path="config", config_name="ippo_cnn_coins")
def main(config):
    """Main entry point."""
    if config.get("TUNE", False):
        # Hyperparameter tuning (not shown for brevity)
        pass
    else:
        single_run(config)


if __name__ == "__main__":
    main()

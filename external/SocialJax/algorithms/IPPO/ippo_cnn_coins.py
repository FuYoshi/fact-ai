"""
Based on PureJaxRL & jaxmarl Implementation of PPO
"""
import sys
sys.path.append('/home/shuqing/SocialJax')
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
# from flax.training import checkpoints
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import socialjax
from socialjax.wrappers.baselines import LogWrapper, SVOLogWrapper
import hydra
from omegaconf import OmegaConf
import wandb
import copy
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import csv
from tensorboardX import SummaryWriter
from datetime import datetime

# FCGrad imports
from fcgrad_utils import fcgrad_adjust, fcgrad_adjust_simple

class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class ActorCritic(nn.Module):
    """
    Actor-Critic network with TWO value heads for FCGrad (paper-faithful).
    
    Paper architecture:
    - Shared CNN encoder (3 conv layers)
    - Shared FC layer (64 units)
    - Actor head: categorical policy
    - V_ind head: individual value estimate (trained on individual rewards)
    - V_col head: collective value estimate (trained on collective rewards)
    
    The two-head design allows computing separate gradients for individual
    and collective objectives, which FCGrad uses for conflict-aware adjustment.
    """
    action_dim: Sequence[int]
    activation: str = "relu"
    use_collective_head: bool = True  # Enable V_col for FCGrad

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(x)

        # Actor head
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # V_ind: Individual value head (standard critic)
        critic_ind = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic_ind = activation(critic_ind)
        critic_ind = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic_ind
        )
        value_ind = jnp.squeeze(critic_ind, axis=-1)

        if self.use_collective_head:
            # V_col: Collective value head (for FCGrad)
            # Paper: "separate value heads sharing the encoder"
            critic_col = nn.Dense(
                64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(embedding)
            critic_col = activation(critic_col)
            critic_col = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
                critic_col
            )
            value_col = jnp.squeeze(critic_col, axis=-1)
            return pi, value_ind, value_col
        else:
            # Backward compatible: return only individual value
            return pi, value_ind


class Transition(NamedTuple):
    """
    Transition tuple for PPO with FCGrad support.
    
    For FCGrad, we store both individual and collective values/rewards
    to compute separate advantages for each objective.
    """
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray           # V_ind: individual value estimate
    value_col: jnp.ndarray       # V_col: collective value estimate (for FCGrad)
    reward: jnp.ndarray          # Individual reward
    reward_col: jnp.ndarray      # Collective reward = mean(all agent rewards)
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def get_rollout(params, config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    use_dual_head = config.get("FCGRAD", False) or config.get("USE_DUAL_HEAD", True)
    if config["PARAMETER_SHARING"]:
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"], use_collective_head=use_dual_head)
    else:
        network = [ActorCritic(env.action_space().n, activation=config["ACTIVATION"], use_collective_head=use_dual_head) for _ in range(env.num_agents)]
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    for o in range(config["GIF_NUM_FRAMES"]):
        print(o)
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)
        if config["PARAMETER_SHARING"]:
            net_out = network.apply(params, obs_batch)
            pi = net_out[0]
            action = pi.sample(seed=key_a0)
            env_act = unbatchify(
                action, env.agents, 1, env.num_agents
            )
        else:
            env_act = {}
            for i in range(env.num_agents):
                net_out = network[i].apply(params[i], obs_batch)
                pi = net_out[0]
                action = pi.sample(seed=key_a0)
                env_act[env.agents[i]] = action




        env_act = {k: v.squeeze() for k, v in env_act.items()}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, env_act)
        done = done["__all__"]

        state_seq.append(state)

    return state_seq


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))

def batchify_dict(x: dict, agent_list, num_actors):
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config, pbar=None, csv_writer=None, tb_writer=None):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    if config["PARAMETER_SHARING"]:
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    else:
        config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env, replace_info=False)

    rew_shaping_anneal = optax.linear_schedule(
        init_value=0.,
        end_value=1.,
        transition_steps=config["REW_SHAPING_HORIZON"],
        transition_begin=config["SHAPING_BEGIN"]
    )

    rew_shaping_anneal_org = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config["REW_SHAPING_HORIZON"],
        transition_begin=config["SHAPING_BEGIN"]
    )
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        # Paper: Always use two-head architecture for fair comparison
        # (even Individual baseline uses same backbone, just ignores V_col)
        use_dual_head = config.get("FCGRAD", False) or config.get("USE_DUAL_HEAD", True)
        
        if config["PARAMETER_SHARING"]:
            network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"], use_collective_head=use_dual_head)
        else:
            network = [ActorCritic(env.action_space().n, activation=config["ACTIVATION"], use_collective_head=use_dual_head) for _ in range(env.num_agents)]

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))

        if config["PARAMETER_SHARING"]:
            network_params = network.init(_rng, init_x)
        else:
            network_params = [network[i].init(_rng, init_x) for i in range(env.num_agents)]
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
        if config["PARAMETER_SHARING"]:
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
        else:
            train_state = [TrainState.create(
                apply_fn=network[i].apply,
                params=network_params[i],
                tx=tx,
            ) for i in range(env.num_agents)]

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)



                # obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(-1, *env.observation_space().shape)

                if config["PARAMETER_SHARING"]:
                    obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                    print("input_obs_shape", obs_batch.shape)
                    net_out = network.apply(train_state.params, obs_batch)
                    if len(net_out) == 3:
                        pi, value, value_col = net_out
                    else:
                        pi, value = net_out
                        value_col = value  # Fallback: use same value for both
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)
                    env_act = unbatchify(
                        action, env.agents, config["NUM_ENVS"], env.num_agents
                    )
                else:
                    obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
                    env_act = {}
                    log_prob = []
                    value = []
                    value_col = []
                    for i in range(env.num_agents):
                        print("input_obs_shape", obs_batch[i].shape)
                        net_out = network[i].apply(train_state[i].params, obs_batch[i])
                        if len(net_out) == 3:
                            pi, value_i, value_col_i = net_out
                        else:
                            pi, value_i = net_out
                            value_col_i = value_i
                        action = pi.sample(seed=_rng)
                        log_prob.append(pi.log_prob(action))
                        env_act[env.agents[i]] = action
                        value.append(value_i)
                        value_col.append(value_col_i)



                # env_act = {k: v.flatten() for k, v in env_act.items()}
                env_act = [v for v in env_act.values()]

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                # current_timestep = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
                # shaped_reward = compute_grouped_rewards(reward)
                # reward = jax.tree_map(lambda x,y: x*rew_shaping_anneal_org(current_timestep)+y*rew_shaping_anneal(current_timestep), reward, shaped_reward)


                # Compute collective reward = mean of all agent rewards
                # Paper: R_col = (1/N) * sum(R^i)
                reward_collective = jnp.mean(reward, axis=1, keepdims=True)  # [num_envs, 1]
                reward_collective = jnp.broadcast_to(reward_collective, reward.shape)  # Same for all agents
                
                if config["PARAMETER_SHARING"]:
                    info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                    transition = Transition(
                        batchify_dict(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                        action,
                        value,
                        value_col,  # V_col estimates
                        batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                        batchify(reward_collective, env.agents, config["NUM_ACTORS"]).squeeze(),  # Collective reward
                        log_prob,
                        obs_batch,
                        info,
                        )
                else:
                    transition = []
                    done = [v for v in done.values()]
                    for i in range(env.num_agents):
                        info_i = {key: jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"]),1), val[:,i]) for key, val in info.items()}
                        transition.append(Transition(
                            done[i],
                            env_act[i],
                            value[i],
                            value_col[i],  # V_col for this agent
                            reward[:,i],
                            reward_collective[:,i],  # Collective reward (same for all agents)
                            log_prob[i],
                            obs_batch[i],
                            info_i,
                        ))
                runner_state = (train_state, env_state, obsv, update_step, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_step, rng = runner_state
            if config["PARAMETER_SHARING"]:
                last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                net_out = network.apply(train_state.params, last_obs_batch)
                if len(net_out) == 3:
                    _, last_val, last_val_col = net_out
                else:
                    _, last_val = net_out
                    last_val_col = last_val
            else:
                last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
                last_val = []
                last_val_col = []
                for i in range(env.num_agents):
                    net_out = network[i].apply(train_state[i].params, last_obs_batch[i])
                    if len(net_out) == 3:
                        _, last_val_i, last_val_col_i = net_out
                    else:
                        _, last_val_i = net_out
                        last_val_col_i = last_val_i
                    last_val.append(last_val_i)
                    last_val_col.append(last_val_col_i)
                last_val = jnp.stack(last_val, axis=0)
                last_val_col = jnp.stack(last_val_col, axis=0)

            def _calculate_gae(traj_batch, last_val, use_collective=False):
                """
                Calculate GAE advantages.
                
                Args:
                    traj_batch: Trajectory batch
                    last_val: Bootstrap value (V_ind or V_col)
                    use_collective: If True, use collective values/rewards for V_col training
                """
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done = transition.done
                    if use_collective:
                        # For V_col: use collective value and collective reward
                        value = transition.value_col
                        reward = transition.reward_col
                    else:
                        # For V_ind: use individual value and individual reward
                        value = transition.value
                        reward = transition.reward
                    
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                if use_collective:
                    return advantages, advantages + traj_batch.value_col
                else:
                    return advantages, advantages + traj_batch.value
            
            # Compute individual advantages (for V_ind)
            if config["PARAMETER_SHARING"]:
                advantages, targets = _calculate_gae(traj_batch, last_val, use_collective=False)
                # Compute collective advantages (for V_col and FCGrad)
                advantages_col, targets_col = _calculate_gae(traj_batch, last_val_col, use_collective=True)
            else:
                advantages = []
                targets = []
                advantages_col = []
                targets_col = []
                for i in range(env.num_agents):
                    # Individual
                    advantages_i, targets_i = _calculate_gae(traj_batch[i], last_val[i], use_collective=False)
                    advantages.append(advantages_i)
                    targets.append(targets_i)
                    # Collective
                    advantages_col_i, targets_col_i = _calculate_gae(traj_batch[i], last_val_col[i], use_collective=True)
                    advantages_col.append(advantages_col_i)
                    targets_col.append(targets_col_i)
                advantages = jnp.stack(advantages, axis=0)
                targets = jnp.stack(targets, axis=0)
                advantages_col = jnp.stack(advantages_col, axis=0)
                targets_col = jnp.stack(targets_col, axis=0)
            # UPDATE NETWORK
            def _update_epoch(update_state, unused, i, all_advantages=None, all_targets_global=None):
                def _update_minbatch(train_state, batch_info, network_used, collective_advantages_minibatch=None, collective_targets_minibatch=None):
                    """
                    Update a single minibatch.

                    Args:
                        train_state: Current training state
                        batch_info: Tuple of (traj_batch, advantages, targets) - SHUFFLED minibatch
                        network_used: Network to use for this agent
                        collective_advantages_minibatch: Collective advantages for THIS minibatch (already shuffled & aligned)
                        collective_targets_minibatch: Collective targets for THIS minibatch (already shuffled & aligned)
                    """
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets, targets_col, network_used):
                        """
                        Loss function for INDIVIDUAL objective.
                        
                        Paper-faithful: trains BOTH value heads:
                        - V_ind on individual targets
                        - V_col on collective targets
                        But actor loss uses individual advantages.
                        """
                        # RERUN NETWORK
                        net_out = network_used.apply(params, traj_batch.obs)
                        if len(net_out) == 3:
                            pi, value_ind, value_col = net_out
                        else:
                            pi, value_ind = net_out
                            value_col = value_ind
                        
                        log_prob = pi.log_prob(traj_batch.action)
                        
                        # CALCULATE VALUE LOSS for V_ind (individual targets)
                        value_pred_clipped = traj_batch.value + (
                            value_ind - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value_ind - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss_ind = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        
                        # CALCULATE VALUE LOSS for V_col (collective targets)
                        value_col_pred_clipped = traj_batch.value_col + (
                            value_col - traj_batch.value_col
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_col_losses = jnp.square(value_col - targets_col)
                        value_col_losses_clipped = jnp.square(value_col_pred_clipped - targets_col)
                        value_loss_col = (
                            0.5 * jnp.maximum(value_col_losses, value_col_losses_clipped).mean()
                        )
                        
                        # Combined value loss (train both heads)
                        value_loss = value_loss_ind + value_loss_col

                        # CALCULATE ACTOR LOSS (using individual advantages)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    def _collective_loss_fn(params, traj_batch, gae, targets, targets_col, network_used, gae_collective=None):
                        """
                        Loss function for COLLECTIVE objective in FCGrad.

                        Paper: R_col = (1/N) * sum(R^i) - the mean return across ALL agents.
                        The collective gradient encourages actions that improve the mean return.
                        
                        Uses V_col head's advantages for actor loss.
                        """
                        # RERUN NETWORK
                        net_out = network_used.apply(params, traj_batch.obs)
                        if len(net_out) == 3:
                            pi, value_ind, value_col = net_out
                        else:
                            pi, value_ind = net_out
                            value_col = value_ind
                        
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS for V_ind
                        value_pred_clipped = traj_batch.value + (
                            value_ind - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value_ind - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss_ind = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        
                        # CALCULATE VALUE LOSS for V_col
                        value_col_pred_clipped = traj_batch.value_col + (
                            value_col - traj_batch.value_col
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_col_losses = jnp.square(value_col - targets_col)
                        value_col_losses_clipped = jnp.square(value_col_pred_clipped - targets_col)
                        value_loss_col = (
                            0.5 * jnp.maximum(value_col_losses, value_col_losses_clipped).mean()
                        )
                        
                        value_loss = value_loss_ind + value_loss_col

                        # CALCULATE ACTOR LOSS with COLLECTIVE advantages (from V_col)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)

                        # Use collective advantages computed from V_col
                        if gae_collective is None:
                            gae_collective = gae

                        # Normalize collective advantages
                        gae_collective = (gae_collective - gae_collective.mean()) / (gae_collective.std() + 1e-8)

                        loss_actor1 = ratio * gae_collective
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae_collective
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    # Get collective targets for this minibatch (for training V_col)
                    targets_col_batch = collective_targets_minibatch if collective_targets_minibatch is not None else targets
                    
                    # Compute individual gradient
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, aux), grads_individual = grad_fn(
                            train_state.params, traj_batch, advantages, targets, targets_col_batch, network_used
                        )

                    # Apply FCGrad if enabled
                    if config.get("FCGRAD", False):
                        # FCGrad: Fair Conflict-aware Gradient Adjustment
                        # Paper: prioritize the objective with LOWER return (disadvantaged)

                        # Individual return: mean of this agent's TD(λ) targets for this minibatch
                        individual_return = jnp.mean(targets)

                        # Collective return: mean of collective targets for this minibatch
                        if collective_targets_minibatch is not None:
                            collective_return = jnp.mean(collective_targets_minibatch)
                        else:
                            collective_return = jnp.mean(targets)

                        # Compute collective gradient using V_col advantages
                        grad_fn_collective = jax.value_and_grad(_collective_loss_fn, has_aux=True)
                        (collective_loss, _), grads_collective = grad_fn_collective(
                            train_state.params, traj_batch, advantages, targets, targets_col_batch, network_used,
                            gae_collective=collective_advantages_minibatch
                        )

                        # Apply FCGrad adjustment
                        # Paper: lower return = disadvantaged = prioritized
                        grads = fcgrad_adjust(
                            grads_individual,
                            grads_collective,
                            individual_return,   # individual return (higher = better)
                            collective_return    # collective return (higher = better)
                        )
                    else:
                        grads = grads_individual

                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (total_loss, aux)

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)

                # Prepare individual batch
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                    )

                # Prepare collective advantages/targets from V_col (passed as all_advantages, all_targets_global)
                # These are now proper V_col-based values, not averaged individual values
                if config.get("FCGRAD", False) and all_advantages is not None:
                    # all_advantages = advantages_col, all_targets_global = targets_col
                    # Shape: [num_agents, num_steps, num_envs] or [num_steps, num_envs] for param sharing
                    if len(all_advantages.shape) == 3:
                        # Non-parameter sharing: use agent i's collective advantages
                        collective_advantages = all_advantages[i]  # [num_steps, num_envs]
                        collective_targets = all_targets_global[i]
                    else:
                        # Parameter sharing: collective is already flat
                        collective_advantages = all_advantages
                        collective_targets = all_targets_global
                    
                    collective_advantages = collective_advantages.reshape((batch_size,))
                    collective_targets = collective_targets.reshape((batch_size,))

                    # Shuffle with SAME permutation as individual data
                    collective_advantages = jnp.take(collective_advantages, permutation, axis=0)
                    collective_targets = jnp.take(collective_targets, permutation, axis=0)

                    # Split into minibatches
                    collective_advantages_minibatches = jnp.reshape(
                        collective_advantages, [config["NUM_MINIBATCHES"], -1]
                    )
                    collective_targets_minibatches = jnp.reshape(
                        collective_targets, [config["NUM_MINIBATCHES"], -1]
                    )
                else:
                    collective_advantages_minibatches = None
                    collective_targets_minibatches = None

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                if config["PARAMETER_SHARING"]:
                    # For parameter sharing, scan over minibatches with None collective data
                    # (FCGrad not fully supported with parameter sharing - need separate handling)
                    train_state, total_loss = jax.lax.scan(
                        lambda state, batch_info: _update_minbatch(state, batch_info, network, None, None),
                        train_state, minibatches
                    )
                else:
                    # FIXED: Scan over minibatches WITH properly aligned collective data
                    if collective_advantages_minibatches is not None:
                        # Combine minibatches with collective data for scanning
                        def scan_fn(state, inputs):
                            batch_info, coll_adv, coll_tgt = inputs
                            return _update_minbatch(state, batch_info, network[i], coll_adv, coll_tgt)

                        # Stack inputs for scan
                        scan_inputs = (minibatches, collective_advantages_minibatches, collective_targets_minibatches)
                        train_state, total_loss = jax.lax.scan(scan_fn, train_state, scan_inputs)
                    else:
                        train_state, total_loss = jax.lax.scan(
                            lambda state, batch_info: _update_minbatch(state, batch_info, network[i], None, None),
                            train_state, minibatches
                        )

                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            if config["PARAMETER_SHARING"]:
                update_state = (train_state, traj_batch, advantages, targets, rng)
                # Pass collective advantages/targets computed from V_col
                update_state, loss_info = jax.lax.scan(
                    lambda state, unused: _update_epoch(state, unused, 0, advantages_col, targets_col), update_state, None, config["UPDATE_EPOCHS"]
                )
                train_state = update_state[0]
                metric = traj_batch.info
                rng = update_state[-1]
            else:
                update_state_dict = []
                metric = []
                for i in range(env.num_agents):
                    update_state = (train_state[i], traj_batch[i], advantages[i], targets[i], rng)
                    # Pass collective advantages/targets computed from V_col (same for all agents)
                    update_state, loss_info = jax.lax.scan(
                        lambda state, unused: _update_epoch(state, unused, i, advantages_col, targets_col), update_state, None, config["UPDATE_EPOCHS"]
                    )
                    update_state_dict.append(update_state)
                    train_state[i] = update_state[0]
                    metric_i = traj_batch[i].info
                    metric_i['loss'] = loss_info[0]
                    metric.append(metric_i)
                    rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)
                # Update progress bar if available
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({
                        'return': f"{float(metric.get('returned_episode_returns', 0)):.2f}",
                        'step': int(metric.get('env_step', 0))
                    })
                # Print periodic status for batch job logs (every 100 updates)
                update = int(metric.get('update_step', 0))
                if update % 100 == 0 or update == 1:
                    r_red = float(metric.get('return_red', 0))
                    r_green = float(metric.get('return_green', 0))
                    print(f"[Update {update}] Step {int(metric.get('env_step', 0)):,} | "
                          f"Return: {float(metric.get('returned_episode_returns', 0)):.2f} | "
                          f"Red: {r_red:.2f} Green: {r_green:.2f}", flush=True)
                # Write to CSV if available
                if csv_writer is not None:
                    csv_writer.writerow([
                        int(metric.get('update_step', 0)),
                        int(metric.get('env_step', 0)),
                        float(metric.get('returned_episode_returns', 0)),
                        float(metric.get('return_red', 0)),
                        float(metric.get('return_green', 0)),
                        float(metric.get('eat_own_coins', 0)),
                        float(metric.get('loss', 0))
                    ])
                # Write to TensorBoard if available
                if tb_writer is not None:
                    step = int(metric.get('env_step', 0))
                    # Main metrics
                    tb_writer.add_scalar('episode/return_mean', float(metric.get('returned_episode_returns', 0)), step)
                    tb_writer.add_scalar('episode/return_red', float(metric.get('return_red', 0)), step)
                    tb_writer.add_scalar('episode/return_green', float(metric.get('return_green', 0)), step)
                    tb_writer.add_scalar('episode/eat_own_coins', float(metric.get('eat_own_coins', 0)), step)
                    tb_writer.add_scalar('episode/length', float(metric.get('returned_episode_lengths', 0)), step)
                    # Fairness metrics (computed on the fly)
                    r_red = float(metric.get('return_red', 0))
                    r_green = float(metric.get('return_green', 0))
                    # Gini: use absolute values to handle negative returns
                    total = abs(r_red) + abs(r_green)
                    if total > 0:
                        gini = abs(r_red - r_green) / (total + 1e-8)
                        # Jain: only meaningful for positive returns
                        if r_red > 0 and r_green > 0:
                            jain = (r_red + r_green)**2 / (2 * (r_red**2 + r_green**2 + 1e-8))
                        else:
                            jain = 0.0  # Unfair if one agent has negative return
                        tb_writer.add_scalar('fairness/gini', gini, step)
                        tb_writer.add_scalar('fairness/jain', jain, step)
                        tb_writer.add_scalar('fairness/min_return', min(r_red, r_green), step)
                    # Training loss (if available)
                    if 'loss' in metric:
                        tb_writer.add_scalar('train/total_loss', float(metric.get('loss', 0)), step)
                    if 'value_loss' in metric:
                        tb_writer.add_scalar('train/value_loss', float(metric.get('value_loss', 0)), step)
                    if 'actor_loss' in metric:
                        tb_writer.add_scalar('train/actor_loss', float(metric.get('actor_loss', 0)), step)
                    if 'entropy' in metric:
                        tb_writer.add_scalar('train/entropy', float(metric.get('entropy', 0)), step)


            update_step = update_step + 1
            
            # Extract per-agent returns BEFORE taking mean (for fairness metrics)
            if config["PARAMETER_SHARING"]:
                # Get raw per-agent returns before mean
                raw_returns = metric["returned_episode_returns"]  # shape: (num_envs, num_agents) or similar
                per_agent_returns = jnp.mean(raw_returns, axis=0)  # mean over envs, keep agents
            else:
                # For non-parameter-sharing, collect from each agent's metric
                per_agent_returns = jnp.array([m["returned_episode_returns"].mean() for m in metric])
            
            metric = jax.tree_map(lambda x: x.mean(), metric)
            if config["PARAMETER_SHARING"]:
                metric["update_step"] = update_step
                metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                # jax.debug.callback(callback, metric)
            else:
                for i in range(env.num_agents):
                    metric[i]["update_step"] = update_step
                    metric[i]["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                metric = metric[0]
                # jax.debug.callback(callback, metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric["eat_own_coins"] = metric["eat_own_coins"] * config["ENV_KWARGS"]["num_inner_steps"]
            
            # Add per-agent returns for fairness analysis
            metric["return_red"] = per_agent_returns[0]    # Agent 0 = Red
            metric["return_green"] = per_agent_returns[1]  # Agent 1 = Green
            
            jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, update_step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

def single_run(config):
    config = OmegaConf.to_container(config)
    # layout_name = copy.deepcopy(config["ENV_KWARGS"]["layout"])
    # config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]

    # Determine algorithm name based on FCGrad setting
    fcgrad_enabled = config.get("FCGRAD", False)
    algo_name = "fcgrad" if fcgrad_enabled else "ippo"
    run_name = f'{algo_name}_cnn_{config["ENV_NAME"]}'

    if fcgrad_enabled:
        print("=" * 60)
        print("FCGrad ENABLED - Fair Conflict-aware Gradient Adjustment")
        print("=" * 60)

    tags = ["FCGRAD" if fcgrad_enabled else "IPPO", "FF"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=tags,
        config=config,
        mode=config["WANDB_MODE"],
        name=run_name
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    
    # Calculate number of updates for progress bar
    num_updates = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    
    print(f"\n🚀 Starting training: {num_updates} updates, {int(config['TOTAL_TIMESTEPS']):,} timesteps")
    print(f"   JIT compiling... (this may take a while with {config['NUM_MINIBATCHES']} minibatches)\n")
    
    # Create progress bar (with settings for batch jobs)
    # mininterval=30 means update at most every 30 seconds (cleaner logs)
    pbar = tqdm(total=num_updates, desc="Training", unit="update", 
                mininterval=30, file=sys.stdout, dynamic_ncols=False)
    
    # Create CSV log file for learning curve (with timestamp to avoid overwriting)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"./logs/{run_name}_seed{config['SEED']}_{timestamp}.csv"
    os.makedirs("./logs", exist_ok=True)
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['update_step', 'env_step', 'returned_episode_returns', 'return_red', 'return_green', 'eat_own_coins', 'loss'])
    
    # Create TensorBoard writer
    tb_log_dir = f"./runs/{run_name}_seed{config['SEED']}_{timestamp}"
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"📈 TensorBoard logs: {tb_log_dir}")
    
    train_jit = jax.jit(make_train(config, pbar=pbar, csv_writer=csv_writer, tb_writer=tb_writer))
    out = jax.vmap(train_jit)(rngs)
    
    pbar.close()
    csv_file.close()
    tb_writer.close()
    print(f"📊 Learning curve saved to: {csv_path}")

    print("** Saving Results **")
    filename = f'{config["ENV_NAME"]}_seed{config["SEED"]}'
    train_state = jax.tree_map(lambda x: x[0], out["runner_state"][0])
    save_path = f"./checkpoints/individual/{filename}.pkl"
    if config["PARAMETER_SHARING"]:
        save_path = f"./checkpoints/indvidual/{filename}.pkl"
        save_params(train_state, save_path)
        params = load_params(save_path)
    else:
        params = []
        for i in range(config['ENV_KWARGS']['num_agents']):
            save_path = f"./checkpoints/individual/{filename}_{i}.pkl"
            save_params(train_state[i], save_path)
            params.append(load_params(save_path))
    evaluate(params, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), save_path, config, timestamp)
    # state_seq = get_rollout(train_state.params, config)
    # viz = OvercookedVisualizer()
    # agent_view_size is hardcoded as it determines the padding around the layout.
    # viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")

def save_params(train_state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)

    with open(save_path, 'wb') as f:
        pickle.dump(params, f)

def load_params(load_path):
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)

def evaluate(params, env, save_path, config, timestamp=None):
    rng = jax.random.PRNGKey(0)

    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    done = False

    pics = []
    img = env.render(state)
    pics.append(img)
    root_dir = f"evaluation/coins"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    for o_t in range(config["GIF_NUM_FRAMES"]):
        # Get observations for all agents
        if config["PARAMETER_SHARING"]:
            obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)
            network = ActorCritic(action_dim=env.action_space().n, activation="relu", use_collective_head=True)
            net_out = network.apply(params, obs_batch)
            pi = net_out[0]  # First output is always policy
            rng, _rng = jax.random.split(rng)
            actions = pi.sample(seed=_rng)
            env_act = {k: v.squeeze() for k, v in unbatchify(
                actions, env.agents, 1, env.num_agents
            ).items()}
        else:
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            env_act = {}
            network = [ActorCritic(action_dim=env.action_space().n, activation="relu", use_collective_head=True) for _ in range(env.num_agents)]
            for i in range(env.num_agents):
                obs_i = jnp.expand_dims(obs_batch[i], axis=0)
                net_out = network[i].apply(params[i], obs_i)
                pi = net_out[0]  # First output is policy
                rng, _rng = jax.random.split(rng)
                single_action = pi.sample(seed=_rng)
                env_act[env.agents[i]] = single_action


        # 执行动作
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])
        done = done["__all__"]

        # 记录结果
        # episode_reward += sum(reward.values())

        # 渲染
        img = env.render(state)
        pics.append(img)

        print('###################')
        print(f'Actions: {env_act}')
        print(f'Reward: {reward}')
        # print(f'State: {state.agent_locs}')
        # print(f'State: {state.claimed_indicator_time_matrix}')
        print("###################")

    # 保存GIF
    print(f"Saving Episode GIF")
    pics = [Image.fromarray(np.array(img)) for img in pics]
    n_agents = len(env.agents)
    timestamp_str = f"_{timestamp}" if timestamp else ""
    gif_path = f"{root_dir}/{n_agents}-agents_seed-{config['SEED']}_frames-{o_t + 1}{timestamp_str}.gif"
    pics[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics[1:],
        duration=200,
        loop=0,
    )

    # Log the GIF to WandB
    print("Logging GIF to WandB")
    wandb.log({"Episode GIF": wandb.Video(gif_path, caption="Evaluation Episode", format="gif")})

        # print(f"Episode {episode} total reward: {episode_reward}")
def tune(default_config):
    """
    Hyperparameter sweep with wandb, including logic to:
    - Initialize wandb
    - Train for each hyperparameter set
    - Save checkpoint
    - Evaluate and log GIF
    """
    import copy

    default_config = OmegaConf.to_container(default_config)

    sweep_config = {
        "name": "coins",
        "method": "grid",
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            # "LR": {"values": [0.001, 0.0005, 0.0001, 0.00005]},
            # "ACTIVATION": {"values": ["relu", "tanh"]},
            # "UPDATE_EPOCHS": {"values": [2, 4, 8]},
            # "NUM_MINIBATCHES": {"values": [4, 8, 16, 32]},
            # "CLIP_EPS": {"values": [0.1, 0.2, 0.3]},
            # "ENT_COEF": {"values": [0.001, 0.01, 0.1]},
            # "NUM_STEPS": {"values": [64, 128, 256]},
            # "ENV_KWARGS.svo_w": {"values": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
            # "ENV_KWARGS.svo_ideal_angle_degrees": {"values": [0, 45, 90]},
            "SEED": {"values": [42, 52, 62]},

        },
    }

    def wrapped_make_train():


        wandb.init(project=default_config["PROJECT"])
        config = copy.deepcopy(default_config)
        # only overwrite the single nested key we're sweeping
        for k, v in dict(wandb.config).items():
            if "." in k:
                parent, child = k.split(".", 1)
                config[parent][child] = v
            else:
                config[k] = v


        # Rename the run for clarity
        run_name = f"sweep_{config['ENV_NAME']}_seed{config['SEED']}"
        wandb.run.name = run_name
        print("Running experiment:", run_name)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))
        train_state = jax.tree_map(lambda x: x[0], outs["runner_state"][0])

        # Evaluate and log
        # params = load_params(train_state.params)
        # test_env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        # evaluate(params, test_env, config)

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(version_base=None, config_path="config", config_name="ippo_cnn_coins")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)
if __name__ == "__main__":
    main()

import functools
from typing import Optional, Dict, Any

import jax
import jax.numpy as jnp
from flax import struct

from brax.envs import base
from brax.envs.base import Env, State

from min_max_rl.networks import ma_po_networks


class Bilinear(Env):
    """Stateless 2-player environment with bilinear reward: r = a1 * a2."""

    def __init__(self):
        self.number_of_players = 2
        self.d = 1  # one action per player

    def reset(self, rng: jax.Array) -> State:
        obs = jnp.array([0.0])  # constant dummy observation
        reward = jnp.zeros((2,))  # both players
        done = jnp.array(False)
        return State(
            pipeline_state=None,
            obs=obs,
            reward=reward,
            done=done,
            metrics={},
            info={},
        )

    def step(self, state: State, action: jax.Array) -> State:
        a1 = action[0]
        a2 = action[1]
        r = a1 * a2
        reward = jnp.array([r, -r])
        done = jnp.array(True)  # ends after one step
        obs = jnp.array([0.0])

        return state.replace(
            obs=obs,
            reward=reward,
            done=done,
        )

    @property
    def observation_size(self) -> int:
        return 1  # scalar dummy observation

    @property
    def action_size(self) -> int:
        return 1  # action size per agent

    @property
    def backend(self) -> str:
        return "none"

    #
    # NETWORK FACTORY
    #

    @property
    def network_factory(self):
        make_network_fn = functools.partial(
            ma_po_networks.make_normal_dist_network,
            bias_init=jax.nn.initializers.constant(1.0)
        )

        return functools.partial(
            ma_po_networks.make_ma_po_networks,
            make_network_fn=make_network_fn
        )

    #
    # HYPERPARAMETERS
    #

    @property
    def dpgda_hps(self):
        return {
            "learning_rate": 1e-4,
            "discounting": 0.9,
            "unroll_length": 1,
            "batch_size": 1,
            "num_minibatches": 2000,
            "num_updates_per_batch": 1,
            "num_resets_per_eval": 0,
            "normalize_observations": False,
            "reward_scaling": 1.0,
            "deterministic_eval": True,
            "restore_checkpoint_path": None,
            "train_step_multiplier": 1,
            # "policy_layers": [],
        }

    @property
    def edpg_hps(self):
        return {
            "learning_rate": 1e-1,
            "discounting": 0.9,
            "unroll_length": 1,
            "batch_size": 1,
            "num_minibatches": 1,
            "num_updates_per_batch": 1,
            "num_resets_per_eval": 0,
            "normalize_observations": False,
            "reward_scaling": 1.0,
            "deterministic_eval": False,
            "restore_checkpoint_path": None,
            "train_step_multiplier": 1,
            "policy_layers": [],
        }

    @property
    def cdpgd_hps(self):
        return {
            "learning_rate": 1e-1,
            "discounting": 0.9,
            "unroll_length": 1,
            "batch_size": 1,
            "num_minibatches": 1,
            "num_updates_per_batch": 1,
            "num_resets_per_eval": 0,
            "normalize_observations": False,
            "reward_scaling": 1.0,
            "deterministic_eval": False,
            "restore_checkpoint_path": None,
            "train_step_multiplier": 1,
            # "policy_layers": [],
        }

    @property
    def cdpgo_hps(self):
        return {
            "learning_rate": 1e-1,
            "alpha": 1.0,
            "discounting": 0.9,
            "unroll_length": 1,
            "batch_size": 1,
            "num_minibatches": 1,
            "num_updates_per_batch": 1,
            "num_resets_per_eval": 0,
            "normalize_observations": False,
            "reward_scaling": 1.0,
            "deterministic_eval": False,
            "restore_checkpoint_path": None,
            "train_step_multiplier": 1,
            # "policy_layers": [],
        }

    @property
    def vpgda_hps(self):
        return {
            "learning_rate": 1e-1,
            "discounting": 0.9,
            "unroll_length": 1,
            "batch_size": 2000,
            "num_minibatches": 1,
            "num_updates_per_batch": 1,
            "num_resets_per_eval": 0,
            "normalize_observations": False,
            "reward_scaling": 1.0,
            "deterministic_eval": False,
            "restore_checkpoint_path": None,
            "train_step_multiplier": 1,
            "policy_layers": [],
        }

    @property
    def evpg_hps(self):
        return {
            "learning_rate": 1e-4,
            "discounting": 0.9,
            "unroll_length": 1,
            "batch_size": 2000,
            "num_minibatches": 1,
            "num_updates_per_batch": 1,
            "num_resets_per_eval": 0,
            "normalize_observations": False,
            "reward_scaling": 1.0,
            "deterministic_eval": False,
            "restore_checkpoint_path": None,
            "train_step_multiplier": 1,
            "policy_layers": [],
        }

    @property
    def cvpgd_hps(self):
        return {
            "learning_rate": 1e-1,
            "discounting": 0.9,
            "unroll_length": 1,
            "batch_size": 2000,
            "num_minibatches": 1,
            "num_updates_per_batch": 1,
            "num_resets_per_eval": 0,
            "normalize_observations": False,
            "reward_scaling": 1.0,
            "deterministic_eval": False,
            "restore_checkpoint_path": None,
            "train_step_multiplier": 1,
            "policy_layers": [],
        }

    @property
    def cvpgo_hps(self):
        return {
            "learning_rate": 1e-1,
            "alpha": 1.0,
            "discounting": 0.9,
            "unroll_length": 1,
            "batch_size": 2000,
            "num_minibatches": 1,
            "num_updates_per_batch": 1,
            "num_resets_per_eval": 0,
            "normalize_observations": False,
            "reward_scaling": 1.0,
            "deterministic_eval": False,
            "restore_checkpoint_path": None,
            "train_step_multiplier": 1,
            "policy_layers": [],
        }

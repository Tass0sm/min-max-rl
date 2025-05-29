from typing import Optional, Dict, Any

import jax
import jax.numpy as jnp
from flax import struct

from brax.envs import base
from brax.envs.base import Env, State


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
    # HYPERPARAMETERS
    #

    @property
    def gda_po_hps(self):
        return {
            "learning_rate": 1e-2,
            "entropy_cost": 1e-2,
            "discounting": 0.9,
            "unroll_length": 1,
            "batch_size": 1,
            "num_minibatches": 1,
            "num_updates_per_batch": 1,
            "num_resets_per_eval": 0,
            "normalize_observations": False,
            "reward_scaling": 1.0,
            "clipping_epsilon": 0.3,
            "gae_lambda": 0.95,
            "deterministic_eval": False,
            "normalize_advantage": True,
            "restore_checkpoint_path": None,
            "train_step_multiplier": 1,
        }

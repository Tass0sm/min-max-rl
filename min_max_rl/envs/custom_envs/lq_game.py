from typing import Optional, Dict, Any

import jax
import jax.numpy as jnp
from flax import struct

from brax.envs import base
from brax.envs.base import Env, State


@struct.dataclass
class LQGameState(State):
    """Custom state for LQ game (no pipeline, multi-agent reward)."""
    # Inherits: pipeline_state, obs, reward, done, metrics, info
    # We use defaults; only obs, reward, and done are updated each step


class LQGame(Env):
    """Linear-Quadratic (LQ) 2-player game with bilinear rewards, stateless dynamics."""

    def __init__(self):
        self.n = 1  # state dimension
        self.d = 1  # action dimension per player

        self.A = jnp.array([[0.9]])
        self.B1 = jnp.array([[0.8]])
        self.B2 = jnp.array([[1.5]])
        self.W12 = jnp.array([[1.0]])
        self.W21 = jnp.array([[1.0]])

    def reset(self, rng: jax.Array) -> LQGameState:
        obs = jnp.zeros((self.n, 1))
        reward = jnp.zeros((2,))  # one reward per player
        done = jnp.array(False)
        return LQGameState(
            pipeline_state=None,
            obs=obs,
            reward=reward,
            done=done,
            metrics={},
            info={},
        )

    def step(self, state: LQGameState, action: jax.Array) -> LQGameState:
        a1 = action[:self.d].reshape((self.d, 1))
        a2 = action[self.d:].reshape((self.d, 1))
        x = state.obs

        x_next = self.A @ x + self.B1 @ a1 - self.B2 @ a2
        r0 = -a1**2 + a2**2 + x_next**2
        reward = jnp.stack([r0.squeeze(), r0.squeeze()])
        done = jnp.array(False)

        return LQGameState(
            pipeline_state=None,
            obs=x_next,
            reward=reward,
            done=done,
            metrics={},
            info={},
        )

    @property
    def observation_size(self) -> int:
        return self.n * 1

    @property
    def action_size(self) -> int:
        return 2 * self.d  # two players

    @property
    def backend(self) -> str:
        return "none"

    #
    # HYPERPARAMETERS
    #

    @property
    def cgd_po_hps(self):
        return {
            "learning_rate": 1e-4,
            "entropy_cost": 1e-4,
            "discounting": 0.9,
            "unroll_length": 10,
            "batch_size": 32,
            "num_minibatches": 16,
            "num_updates_per_batch": 2,
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

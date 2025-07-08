from typing import Optional, Dict, Any
import jax
import jax.numpy as jnp
from flax import struct
from brax.envs import base
from brax.envs.base import Env, State


class CarRacing(Env):
    """Two-player car racing environment using blended dynamics."""

    def __init__(self):
        self.n_players = 2
        self.n_state = 6
        self.n_control = 2

        self.Ts = 0.03
        self.mass = 0.041
        self.mass_long = 0.041
        self.l_f = 0.029
        self.l_r = 0.033
        self.I_z = 27.8e-6

        self.Cm1 = 0.287
        self.Cm2 = 0.054527
        self.Cr0 = 0.051891
        self.Cr2 = 0.000348

        self.B_r = 3.3852 / 1.2
        self.C_r = 1.2691
        self.D_r = 0.1737 * 1.2

        self.B_f = 2.579
        self.C_f = 1.2
        self.D_f = 1.05 * 0.192

    def reset(self, rng: jax.Array) -> State:
        obs = jnp.concatenate([
            jnp.array([0., -0.1, 0., 0.4, 0., 0.]),
            jnp.array([0.,  0.1, 0., 0.4, 0., 0.])
        ])
        reward = jnp.zeros((2,))
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
        # action: shape (2, 2) for two players
        x_prev = state.obs.reshape(-1, self.n_state)
        u = jnp.clip(action, -1.0, 1.0).reshape(-1, self.n_control)

        def single_step(x, u):
            vx = x[3]
            lambda_blend = jnp.clip((vx - 0.3) / 0.2, 0.0, 1.0)
            x_dyn = self._rk4(self._dx_dynamic, x, u)
            x_kin = self._rk4(self._dx_kinematic, self._to_kinematic(x), u)
            x_kin_full = self._from_kinematic(x_kin, u)
            return lambda_blend * x_dyn + (1 - lambda_blend) * x_kin_full

        x_next = jax.vmap(single_step)(x_prev, u)

        # Compute rewards based on relative progress
        delta_s = x_next[:, 0] - x_prev[:, 0]
        base_reward = delta_s[0] - delta_s[1]
        r1 = base_reward + jnp.where(x_next[0, 0] > 20.0, 1.0, 0.0)
        r2 = -base_reward + jnp.where(x_next[1, 0] > 20.0, 1.0, 0.0)
        reward = jnp.array([r1, r2])

        # Done condition: either car reached the finish line
        done = jnp.any(x_next[:, 0] > 20.0)

        return state.replace(
            obs=x_next.flatten(),
            reward=reward,
            done=done,
        )

    def _dx_dynamic(self, x, u):
        s, d, mu, vx, vy, r = x
        a, delta = u

        alpha_f = -jnp.arctan((self.l_f * r + vy) / (vx + 1e-5)) + delta
        alpha_r = jnp.arctan((self.l_r * r - vy) / (vx + 1e-5))

        F_rx = self.Cm1 * a - self.Cm2 * vx * a - self.Cr2 * vx**2 - self.Cr0
        F_ry = self.D_r * jnp.sin(self.C_r * jnp.arctan(self.B_r * alpha_r))
        F_fy = self.D_f * jnp.sin(self.C_f * jnp.arctan(self.B_f * alpha_f))

        dx = jnp.zeros(6)
        dx = dx.at[0].set(vx * jnp.cos(mu) - vy * jnp.sin(mu))
        dx = dx.at[1].set(vx * jnp.sin(mu) + vy * jnp.cos(mu))
        dx = dx.at[2].set(r)
        dx = dx.at[3].set((F_rx - F_fy * jnp.sin(delta) + self.mass * vy * r) / self.mass_long)
        dx = dx.at[4].set((F_ry + F_fy * jnp.cos(delta) - self.mass * vx * r) / self.mass)
        dx = dx.at[5].set((F_fy * self.l_f * jnp.cos(delta) - F_ry * self.l_r) / self.I_z)
        return dx

    def _dx_kinematic(self, x, u):
        s, d, mu, v = x
        a, delta = u
        beta = jnp.arctan(self.l_r * jnp.tan(delta) / (self.l_f + self.l_r))

        D = jnp.where(
            v <= 0.1,
            jnp.maximum((self.Cr0 + self.Cr2 * v ** 2) / (self.Cm1 - self.Cm2 * v), a),
            a,
        )

        ds = v * jnp.cos(beta + mu)
        dd = v * jnp.sin(beta + mu)
        dmu = v * jnp.sin(beta) / self.l_r
        dv = (self.Cm1 * D - self.Cm2 * v * D - self.Cr0 - self.Cr2 * v ** 2) / self.mass_long

        return jnp.array([ds, dd, dmu, dv])

    def _to_kinematic(self, x):
        v = jnp.linalg.norm(x[3:5])
        return jnp.array([x[0], x[1], x[2], v])

    def _from_kinematic(self, kin, u):
        s, d, mu, v = kin
        delta = u[1]
        beta = jnp.arctan(self.l_r * jnp.tan(delta) / (self.l_f + self.l_r))

        vx = v * jnp.cos(beta)
        vy = v * jnp.sin(beta)
        r = vx * jnp.tan(delta) / (self.l_f + self.l_r)

        return jnp.array([s, d, mu, vx, vy, r])

    def _rk4(self, f, x, u):
        k1 = f(x, u)
        k2 = f(x + self.Ts / 2 * k1, u)
        k3 = f(x + self.Ts / 2 * k2, u)
        k4 = f(x + self.Ts * k3, u)
        return x + self.Ts * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    @property
    def observation_size(self) -> int:
        return self.n_state

    @property
    def action_size(self) -> int:
        return self.n_control

    @property
    def backend(self) -> str:
        return "none"

    #
    # HYPERPARAMETERS
    #

    @property
    def vpgda_hps(self):
        return {
            "learning_rate": 1e-2,
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
        }

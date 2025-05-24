import jax
import jax.numpy as jnp
import jax.random as random
import chex
from flax import struct
from typing import Tuple, Optional, Dict, Any
from gymnax.environments.environment import Environment
from gymnax.environments.spaces import Box
from gymnax.environments import spaces



@struct.dataclass
class EnvState:
    state: chex.Array  # shape (n, 1)
    done: bool


@struct.dataclass
class EnvParams:
    A: chex.Array  # shape (n, n)
    B1: chex.Array  # shape (n, d)
    B2: chex.Array  # shape (n, d)
    W12: chex.Array  # shape (d, d)
    W21: chex.Array  # shape (d, d)


class LQGame(Environment):
    def __init__(self):
        super().__init__()
        self.n = 1
        self.d = 1

    def default_params(self) -> EnvParams:
        return EnvParams(
            A=jnp.array([[0.9]]),
            B1=jnp.array([[0.8]]),
            B2=jnp.array([[1.5]]),
            W12=jnp.array([[1.0]]),
            W21=jnp.array([[1.0]])
        )

    def reset(
        self, rng: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        init_state = jnp.zeros((self.n, 1))
        return init_state, EnvState(state=init_state, done=False)

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, chex.Array, bool, Dict[str, Any]]:
        # Split 2-player action into tuple
        a1 = action[..., :self.d].reshape((self.d, 1))
        a2 = action[..., self.d:].reshape((self.d, 1))

        # Dynamics update
        s_next = params.A @ state.state + params.B1 @ a1 - params.B2 @ a2

        # Reward (same for both players here, could be extended)
        r0 = -a1**2 + a2**2 + s_next**2
        reward = jnp.concatenate([r0, r0], axis=0).squeeze()  # shape (2,)

        # No terminal condition in this setup
        done = False
        next_state = EnvState(state=s_next, done=done)
        return s_next, next_state, reward, done, {}

    def observation_space(self, params: EnvParams) -> Box:
        return Box(low=-jnp.inf, high=jnp.inf, shape=(self.n, 1), dtype=jnp.float32)

    def action_space(self, params: EnvParams) -> Box:
        # Both players act: concatenate their actions
        return Box(low=-jnp.inf, high=jnp.inf, shape=(2 * self.d,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> Box:
        return Box(low=-jnp.inf, high=jnp.inf, shape=(self.n, 1), dtype=jnp.float32)


if __name__ == "__main__":
    env = LQGameGymnax()
    params = env.default_params()
    rng = random.PRNGKey(0)
    obs, state = env.reset(rng, params)

    action = jnp.array([0.1, -0.2])  # (action1, action2)
    obs, next_state, reward, done, _ = env.step_env(rng, state, action, params)

    print("Obs:", obs)
    print("Reward:", reward)

import jax
from brax.envs import PipelineEnv, State, Wrapper
from jax import numpy as jnp


class TrajectoryIdWrapper(Wrapper):
    def __init__(self, env: PipelineEnv, key_name = "traj_id"):
        super().__init__(env)
        self._key_name = key_name

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info[self._key_name] = jnp.zeros(rng.shape[:-1])
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info.keys():
            key = state.info[self._key_name] + jnp.where(state.info["steps"], 0, 1)
        else:
            key = state.info[self._key_name]
        state = self.env.step(state, action)
        state.info[self._key_name] = key
        return state

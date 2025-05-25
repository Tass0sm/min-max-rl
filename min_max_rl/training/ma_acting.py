"""Multi-agent acting functions."""

import time
from typing import Callable, Sequence, Tuple

from brax import envs
from brax.training.types import Metrics
from brax.training.types import Policy
from brax.training.types import PolicyParams
from brax.training.types import PRNGKey
from brax.training.types import Transition
import jax
import numpy as np

State = envs.State
Env = envs.Env


def ma_actor_step(
    env: Env,
    env_state: State,
    policies: list[Policy],
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition]:
  """Collect data."""
  actions, policy_extras = policy(env_state.obs, key)
  nstate = env.step(env_state, actions)
  state_extras = {x: nstate.info[x] for x in extra_fields}
  return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=env_state.obs,
      action=actions,
      reward=nstate.reward,
      discount=1 - nstate.done,
      next_observation=nstate.obs,
      extras={'policy_extras': policy_extras, 'state_extras': state_extras},
  )


def ma_generate_unroll(
    env: Env,
    env_state: State,
    policies: list[Policy],
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = (),
    return_states: bool = False,
) -> Tuple[State, Transition]:
  """Collect trajectories of given unroll_length."""

  @jax.jit
  def f(carry, unused_t):
    state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, transition = actor_step(
        env, state, policy, current_key, extra_fields=extra_fields
    )
    if return_states:
      return (nstate, next_key), (state, transition)
    else:
      return (nstate, next_key), transition

  (final_state, _), data = jax.lax.scan(
      f, (env_state, key), (), length=unroll_length
  )
  return final_state, data

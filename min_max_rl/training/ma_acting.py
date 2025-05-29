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
import jax.numpy as jnp


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
  agent_actions_l = []
  ma_agent_extras = {}
  for i, policy in enumerate(policies):
    actions, policy_extras = policy(env_state.obs, key)
    agent_actions_l.append(actions)
    ma_agent_extras |= {f"agent{i}_{k}":v for k,v in policy_extras.items()}
  ma_actions = jnp.concatenate(agent_actions_l, axis=-1)
  nstate = env.step(env_state, ma_actions)
  state_extras = {x: nstate.info[x] for x in extra_fields}
  return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=env_state.obs,
      action=ma_actions,
      reward=nstate.reward,
      discount=1 - nstate.done,
      next_observation=nstate.obs,
      extras={'ma_agent_extras': ma_agent_extras, 'state_extras': state_extras},
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
    nstate, transition = ma_actor_step(
        env, state, policies, current_key, extra_fields=extra_fields
    )
    if return_states:
      return (nstate, next_key), (state, transition)
    else:
      return (nstate, next_key), transition

  (final_state, _), data = jax.lax.scan(
      f, (env_state, key), (), length=unroll_length
  )
  return final_state, data

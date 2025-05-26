"""Wrappers to support Brax training."""

from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct
import jax
from jax import numpy as jp

from brax.envs.wrappers.training import EvalMetrics


class MAEpisodeWrapper(Wrapper):
  """Maintains episode step count and sets done at episode end. Includes small
  modifications for rewards per agent."""

  def __init__(self, env: Env, episode_length: int, action_repeat: int):
    super().__init__(env)
    self.episode_length = episode_length
    self.action_repeat = action_repeat

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    state.info['steps'] = jp.zeros(rng.shape[:-1], dtype=jp.int32)
    state.info['truncation'] = jp.zeros(rng.shape[:-1], dtype=jp.bool)
    # Keep separate record of episode done as state.info['done'] can be erased
    # by AutoResetWrapper
    state.info['episode_done'] = jp.zeros(rng.shape[:-1], dtype=jp.bool)
    episode_metrics = dict()
    episode_metrics['sum_rewards'] = jp.zeros((*rng.shape[:-1], 2))
    episode_metrics['length'] = jp.zeros(rng.shape[:-1])
    for metric_name in state.metrics.keys():
      episode_metrics[metric_name] = jp.zeros(rng.shape[:-1])
    state.info['episode_metrics'] = episode_metrics
    return state

  def step(self, state: State, action: jax.Array) -> State:
    def f(state, _):
      nstate = self.env.step(state, action)
      return nstate, nstate.reward

    state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
    state = state.replace(reward=jp.sum(rewards, axis=0))
    steps = state.info['steps'] + self.action_repeat
    one = jp.ones_like(state.done)
    zero = jp.zeros_like(state.done)
    episode_length = jp.array(self.episode_length, dtype=jp.int32)
    done = jp.where(steps >= episode_length, one, state.done)
    state.info['truncation'] = jp.where(
        steps >= episode_length, 1 - state.done, zero
    ).astype(jp.bool)
    state.info['steps'] = steps

    # Aggregate state metrics into episode metrics
    prev_done = state.info['episode_done']
    state.info['episode_metrics']['sum_rewards'] += jp.sum(rewards, axis=0)
    state.info['episode_metrics']['sum_rewards'] *= (1 - jp.expand_dims(prev_done, axis=-1))
    state.info['episode_metrics']['length'] += self.action_repeat
    state.info['episode_metrics']['length'] *= (1 - prev_done)
    for metric_name in state.metrics.keys():
      if metric_name != 'rewards':
        state.info['episode_metrics'][metric_name] += state.metrics[metric_name]
        state.info['episode_metrics'][metric_name] *= (1 - prev_done)
    state.info['episode_done'] = done
    return state.replace(done=done)


class MAEvalWrapper(Wrapper):
  """Brax env with eval metrics."""

  def reset(self, rng: jax.Array) -> State:
    reset_state = self.env.reset(rng)
    reset_state.metrics['rewards'] = reset_state.reward
    eval_metrics = EvalMetrics(
        episode_metrics=jax.tree_util.tree_map(
            jp.zeros_like, reset_state.metrics
        ),
        active_episodes=jp.ones_like(reset_state.done, dtype=jp.bool),
        episode_steps=jp.zeros_like(reset_state.done, dtype=jp.int32),
    )
    reset_state.info['eval_metrics'] = eval_metrics
    return reset_state

  def step(self, state: State, action: jax.Array) -> State:
    state_metrics = state.info['eval_metrics']
    if not isinstance(state_metrics, EvalMetrics):
      raise ValueError(
          f'Incorrect type for state_metrics: {type(state_metrics)}'
      )
    del state.info['eval_metrics']
    nstate = self.env.step(state, action)
    nstate.metrics['rewards'] = nstate.reward
    episode_steps = jp.where(
        state_metrics.active_episodes,
        nstate.info['steps'],
        state_metrics.episode_steps,
    )

    def sum_metrics(a, b):
      if b.ndim == 2:
        return a + b * jp.expand_dims(state_metrics.active_episodes, -1)
      else:
        return a + b * state_metrics.active_episodes
    
    episode_metrics = jax.tree_util.tree_map(
        sum_metrics,
        state_metrics.episode_metrics,
        nstate.metrics,
    )
    active_episodes = state_metrics.active_episodes * (1 - nstate.done).astype(jp.bool)

    eval_metrics = EvalMetrics(
        episode_metrics=episode_metrics,
        active_episodes=active_episodes,
        episode_steps=episode_steps,
    )
    nstate.info['eval_metrics'] = eval_metrics
    return nstate

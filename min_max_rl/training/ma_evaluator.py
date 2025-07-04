"""Multi-Agent Evaluator."""

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

from min_max_rl.training.ma_acting import ma_generate_unroll
from min_max_rl.envs.wrappers.ma_training_wrappers import MAEvalWrapper


State = envs.State


class MultiAgentEvaluator:
  """Class to run evaluations."""

  def __init__(
      self,
      eval_env: envs.Env,
      eval_policy_fn: Callable[[PolicyParams], Policy],
      num_eval_envs: int,
      episode_length: int,
      action_repeat: int,
      key: PRNGKey,
  ):
    """Init.

    Args:
      eval_env: Batched environment to run evals on.
      eval_policy_fn: Function returning the policy from the policy parameters.
      num_eval_envs: Each env will run 1 episode in parallel for each eval.
      episode_length: Maximum length of an episode.
      action_repeat: Number of physics steps per env step.
      key: RNG key.
    """
    self._key = key
    self._eval_walltime = 0.0

    eval_env = MAEvalWrapper(eval_env)

    def generate_eval_unroll(
        policy_params: PolicyParams, key: PRNGKey
    ) -> State:
      reset_keys = jax.random.split(key, num_eval_envs)
      eval_first_state = eval_env.reset(reset_keys)
      return ma_generate_unroll(
          eval_env,
          eval_first_state,
          eval_policy_fn(policy_params),
          key,
          unroll_length=episode_length // action_repeat,
      )[0]

    self._generate_eval_unroll = jax.jit(generate_eval_unroll)
    self._steps_per_unroll = episode_length * num_eval_envs

  def run_evaluation(
      self,
      policy_params: PolicyParams,
      training_metrics: Metrics,
      aggregate_episodes: bool = True,
  ) -> Metrics:
    """Run one epoch of evaluation."""
    self._key, unroll_key = jax.random.split(self._key)

    t = time.time()
    eval_state = self._generate_eval_unroll(policy_params, unroll_key)
    eval_metrics = eval_state.info['eval_metrics']
    eval_metrics.active_episodes.block_until_ready()
    epoch_eval_time = time.time() - t
    metrics = {}
    for fn in [np.mean, np.std]:
      suffix = '_std' if fn == np.std else ''
      metrics.update({
          f'eval/episode_{name}{suffix}': (
              fn(value) if aggregate_episodes else value
          )
          for name, value in eval_metrics.episode_metrics.items()
      })

    # overwrite with reward computation
    # TODO: potentially log individual rewards per agent
    rewards = eval_metrics.episode_metrics["rewards"]
    positive_rewards = rewards[..., 0]
    metrics.update({
        'eval/episode_reward': (np.mean(positive_rewards) if aggregate_episodes else positive_rewards),
        'eval/episode_reward_std': (np.std(positive_rewards) if aggregate_episodes else positive_rewards)
    })

    metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
    metrics['eval/std_episode_length'] = np.std(eval_metrics.episode_steps)
    metrics['eval/epoch_eval_time'] = epoch_eval_time
    metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime = self._eval_walltime + epoch_eval_time
    metrics = {
        'eval/walltime': self._eval_walltime,
        **training_metrics,
        **metrics,
    }

    return metrics  # pytype: disable=bad-return-type  # jax-ndarray

"""Competitive Deterministic Policy Gradient Descent
"""

import functools
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import acting, pmap, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params, PRNGKey
from brax.v1 import envs as envs_v1
from etils import epath
from orbax import checkpoint as ocp

from min_max_rl.sma import simple_moving_average
from min_max_rl.networks import ma_po_networks, ma_deterministic_po_networks
from min_max_rl.training import ma_acting, ma_gradients
from min_max_rl.training.ma_evaluator import MultiAgentEvaluator
from min_max_rl.envs.wrappers import TrajectoryIdWrapper, wrap_for_training


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class PONetworkParams:
    """Contains training state for the learner."""
    policy: Params


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    cgd_state: ma_gradients.CGDState
    agent_params: list[PONetworkParams]
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    # brax user code is sometimes ambiguous about weak_type.  in order to
    # avoid extra jit recompilations we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def compute_deterministic_linear_obj(
        params_a: PONetworkParams,
        params_b: PONetworkParams,
        normalizer_params: any,
        data: types.Transition,
        network_a: ppo_networks.PPONetworks,
        network_b: ppo_networks.PPONetworks,
        reward_scaling: float = 1.0,
        agent_idx: int = 0,
) -> Tuple[jnp.ndarray, types.Metrics]:
  """Computes PPO loss.

  Args:
    params: Network parameters,
    normalizer_params: Parameters of the normalizer.
    data: Transition that with leading dimension [B, T]. extra fields required
      are ['policy_extras']['agent{i}_raw_action']
    network: PO networks.
    reward_scaling: reward multiplier.

  Returns:
    A tuple (loss, metrics)
  """

  agent_a_action = network_a.policy_network.apply(normalizer_params, params_a.policy, data.observation)
  agent_b_action = network_b.policy_network.apply(normalizer_params, params_b.policy, data.observation)

  linear_obj = (agent_a_action * agent_b_action).mean()

  return linear_obj, {
      f'linear_obj{agent_idx}': linear_obj,
  }


def compute_deterministic_bilinear_obj(
    params: list[PONetworkParams],
    normalizer_params: any,
    data: types.Transition,
    networks: list[ppo_networks.PPONetworks],
    reward_scaling: float = 1.0,
) -> Tuple[jnp.ndarray, types.Metrics]:
  """Computes PPO loss.

  Args:
    params: list[PO network parameters],
    normalizer_params: Parameters of the normalizer.
    data: Transition that with leading dimension [B, T]. extra fields required
      are ['policy_extras']['raw_action']
    networks: list[PO networks].
    reward_scaling: reward multiplier.

  Returns:
    A tuple (loss, metrics)
  """

  def closed_form_action_value_function(state, agent_0_action, agent_1_action):
      # order doesn't matter and horizon = 1
      return agent_0_action[0] * agent_1_action[0]

  agent_0_action = networks[0].policy_network.apply(normalizer_params, params[0].policy, data.observation)
  agent_1_action = networks[1].policy_network.apply(normalizer_params, params[1].policy, data.observation)

  bilinear_obj_1 = jax.vmap(closed_form_action_value_function)(data.observation, agent_0_action, agent_1_action).mean()
  bilinear_obj_2 = 0.0
  bilinear_obj_3 = 0.0

  bilinear_obj = bilinear_obj_1 + bilinear_obj_2 + bilinear_obj_3

  return bilinear_obj, {
      'bilinear_obj': bilinear_obj,
  }


@dataclass
class CDPGD:
    """Competitive Deterministic Policy Gradient Descent
    Args:
      learning_rate: learning rate for ppo loss
      discounting: discounting rate
      unroll_length: the number of timesteps to unroll in each train_env. The
        PPO loss is computed over `unroll_length` timesteps
      batch_size: the batch size for each minibatch SGD step
      num_minibatches: the number of times to run the SGD step, each with a
        different minibatch with leading dimension of `batch_size`
      num_updates_per_batch: the number of times to run the gradient update over
        all minibatches before doing a new train_env rollout
      num_resets_per_eval: the number of train_env resets to run between each
        eval. The train_env resets occur on the host
      normalize_observations: whether to normalize observations
      reward_scaling: float scaling for reward
      deterministic_eval: whether to run the eval with a deterministic policy
      network_factory: function that generates networks for policy and value
        functions
      progress_fn: a user-defined callback function for reporting/plotting metrics
      restore_checkpoint_path: the path used to restore previous model params
    """

    learning_rate: float = 1e-4
    discounting: float = 0.9
    unroll_length: int = 10
    batch_size: int = 32
    num_minibatches: int = 16
    num_updates_per_batch: int = 2
    num_resets_per_eval: int = 0
    normalize_observations: bool = False
    reward_scaling: float = 1.0
    deterministic_eval: bool = False
    restore_checkpoint_path: Optional[str] = None
    train_step_multiplier: int = 1

    def train_fn(
        self,
        config,
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        """GDA-PO training.

        Args:
          train_env: the train_env to train
          eval_env: an optional train_env for eval only, defaults to `train_env`
          randomization_fn: a user-defined callback function that generates randomized environments
          progress_fn: a user-defined callback function for reporting/plotting metrics

        Returns:
          Tuple of (make_policies function, network params, metrics)
        """
        assert self.batch_size * self.num_minibatches % config.num_envs == 0
        xt = time.time()
        network_factory = ma_deterministic_po_networks.make_ma_deterministic_po_networks

        process_count = jax.process_count()
        process_id = jax.process_index()
        local_device_count = jax.local_device_count()
        local_devices_to_use = local_device_count
        if config.max_devices_per_host:
            local_devices_to_use = min(local_devices_to_use, config.max_devices_per_host)

        logging.info(
            "Device count: %d, process count: %d (id %d), local device count: %d, devices to be used count: %d",
            jax.device_count(),
            process_count,
            process_id,
            local_device_count,
            local_devices_to_use,
        )
        device_count = local_devices_to_use * process_count

        # The number of train_env steps executed for every training step.
        utd_ratio = self.batch_size * self.unroll_length * self.num_minibatches * config.action_repeat
        num_evals_after_init = max(config.num_evals - 1, 1)
        # The number of training_step calls per training_epoch call.
        # equals to ceil(total_env_steps / (num_evals * utd_ratio *
        #                                 num_resets_per_eval))
        num_training_steps_per_epoch = np.ceil(
            config.total_env_steps / (num_evals_after_init * utd_ratio * max(self.num_resets_per_eval, 1))
        ).astype(int)

        key = jax.random.PRNGKey(config.seed)
        global_key, local_key = jax.random.split(key)
        del key
        local_key = jax.random.fold_in(local_key, process_id)
        local_key, key_env, eval_key = jax.random.split(local_key, 3)
        # key_networks should be global, so that networks are initialized the same
        # way for different processes.
        key_policy, key_value = jax.random.split(global_key)
        del global_key

        assert config.num_envs % device_count == 0

        v_randomization_fn = None
        if randomization_fn is not None:
            randomization_batch_size = config.num_envs // local_device_count
            # all devices gets the same randomization rng
            randomization_rng = jax.random.split(key_env, randomization_batch_size)
            v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)

        env = train_env
        env = TrajectoryIdWrapper(env)
        env = wrap_for_training(
            train_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            randomization_fn=v_randomization_fn,
        )
        unwrapped_env = train_env

        reset_fn = jax.jit(jax.vmap(env.reset))
        key_envs = jax.random.split(key_env, config.num_envs // process_count)
        key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
        env_state = reset_fn(key_envs)

        normalize = lambda x, y: x
        if self.normalize_observations:
            normalize = running_statistics.normalize

        num_agents = 2

        networks = network_factory(
            num_agents,
            env_state.obs.shape[-1],
            env.action_size,
            preprocess_observations_fn=normalize,
            policy_hidden_layer_sizes=[],
        )
        make_policies = ma_deterministic_po_networks.make_inference_fns(networks)

        optimizer = optax.sgd(learning_rate=self.learning_rate)

        agent0_linear_obj_term = functools.partial(
            compute_deterministic_linear_obj,
            network_a=networks[0],
            network_b=networks[1],
            reward_scaling=self.reward_scaling,
            agent_idx=0,
        )

        agent1_linear_obj_term = functools.partial(
            compute_deterministic_linear_obj,
            network_a=networks[1],
            network_b=networks[0],
            reward_scaling=self.reward_scaling,
            agent_idx=1,
        )

        bilinear_obj_term = functools.partial(
            compute_deterministic_bilinear_obj,
            networks=networks,
            reward_scaling=self.reward_scaling,
        )

        gradient_update_fn = ma_gradients.cgd_update_fn(
            agent0_linear_obj_term,
            agent1_linear_obj_term,
            bilinear_obj_term,
            optimizer,
            self.learning_rate,
            pmap_axis_name=_PMAP_AXIS_NAME,
            have_aux=True,
        )

        def minibatch_step(
            carry,
            data: types.Transition,
            normalizer_params: running_statistics.RunningStatisticsState,
        ):
            optimizer_state, cgd_state, params, key = carry
            (_, metrics), params, optimizer_state = gradient_update_fn(
                params,
                normalizer_params,
                data,
                optimizer_state=optimizer_state,
                cgo_state=cgd_state
            )

            return (optimizer_state, cgd_state,  params, key), metrics

        def update_step(
            carry,
            unused_t,
            data: types.Transition,
            normalizer_params: running_statistics.RunningStatisticsState,
        ):
            optimizer_state, cgd_state, params, key = carry
            key, key_perm, key_grad = jax.random.split(key, 3)

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(key_perm, x)
                x = jnp.reshape(x, (self.num_minibatches, -1) + x.shape[1:])
                return x

            shuffled_data = jax.tree_util.tree_map(convert_data, data)
            (optimizer_state, cgd_state, params, _), metrics = jax.lax.scan(
                functools.partial(minibatch_step, normalizer_params=normalizer_params),
                (optimizer_state, cgd_state, params, key_grad),
                shuffled_data,
                length=self.num_minibatches,
            )
            return (optimizer_state, cgd_state, params, key), metrics

        def training_step(
            carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t
        ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
            training_state, state, key = carry
            update_key, key_generate_unroll, new_key = jax.random.split(key, 3)

            policies = make_policies((training_state.normalizer_params, [
                ap.policy for ap in training_state.agent_params
            ]))

            def f(carry, unused_t):
                current_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                next_state, data = ma_acting.ma_generate_unroll(
                    env,
                    current_state,
                    policies,
                    current_key,
                    self.unroll_length,
                    extra_fields=("truncation",),
                )
                return (next_state, next_key), data

            (state, _), data = jax.lax.scan(
                f,
                (state, key_generate_unroll),
                (),
                length=self.batch_size * self.num_minibatches // config.num_envs,
            )
            # Have leading dimensions (batch_size * num_minibatches, unroll_length)
            data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
            data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)
            assert data.discount.shape[1:] == (self.unroll_length,)

            # Update normalization params and normalize observations.
            normalizer_params = running_statistics.update(
                training_state.normalizer_params,
                data.observation,
                pmap_axis_name=_PMAP_AXIS_NAME,
            )

            (optimizer_state, cgd_state, agent_params, _), metrics = jax.lax.scan(
                functools.partial(
                    update_step,
                    data=data,
                    normalizer_params=normalizer_params,
                ),
                (
                    training_state.optimizer_state,
                    training_state.cgd_state,
                    training_state.agent_params,
                    update_key,
                ),
                (),
                length=self.num_updates_per_batch,
            )

            new_training_state = TrainingState(
                optimizer_state=optimizer_state,
                cgd_state=cgd_state,
                agent_params=agent_params,
                normalizer_params=normalizer_params,
                env_steps=training_state.env_steps + utd_ratio,
            )
            return (new_training_state, state, new_key), metrics

        def training_epoch(
            training_state: TrainingState,
            state: envs.State,
            key: PRNGKey,
        ) -> Tuple[TrainingState, envs.State, Metrics]:
            (training_state, state, _), loss_metrics = jax.lax.scan(
                training_step,
                (training_state, state, key),
                (),
                length=num_training_steps_per_epoch,
            )
            loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
            return training_state, state, loss_metrics

        training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

        # Note that this is NOT a pure jittable method.
        def training_epoch_with_timing(
            training_state: TrainingState,
            env_state: envs.State,
            key: PRNGKey,
        ) -> Tuple[TrainingState, envs.State, Metrics]:
            nonlocal training_walltime
            t = time.time()
            training_state, env_state = _strip_weak_type((training_state, env_state))
            result = training_epoch(training_state, env_state, key)
            training_state, env_state, metrics = _strip_weak_type(result)

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (
                num_training_steps_per_epoch * utd_ratio * max(self.num_resets_per_eval, 1)
            ) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            return (
                training_state,
                env_state,
                metrics,
            )  # pytype: disable=bad-return-type  # py311-upgrade

        # Initialize model params and training state.
        policy_keys = jax.random.split(key_policy, num_agents)
        value_keys = jax.random.split(key_value, num_agents)
        init_params = [
            PONetworkParams(policy=network.policy_network.init(policy_key))
            for network, policy_key, value_key in zip(networks, policy_keys, value_keys)
        ]

        training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
            optimizer_state=optimizer.init(init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
            cgd_state=ma_gradients.cgd_init_state(init_params),
            agent_params=init_params,
            normalizer_params=running_statistics.init_state(
                specs.Array(env_state.obs.shape[-1:], jnp.dtype("float32"))
            ),
            env_steps=0,
        )

        if config.total_env_steps == 0:
            return (
                make_policies,
                (
                    training_state.normalizer_params,
                    [ap.policy for ap in training_state.agent_params],
                ),
                {},
            )

        if self.restore_checkpoint_path is not None and epath.Path(self.restore_checkpoint_path).exists():
            logging.info(
                "restoring from checkpoint %s",
                self.restore_checkpoint_path,
            )
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            target = training_state.normalizer_params, init_params
            (normalizer_params, init_params) = orbax_checkpointer.restore(
                self.restore_checkpoint_path, item=target
            )
            training_state = training_state.replace(normalizer_params=normalizer_params, params=init_params)

        training_state = jax.device_put_replicated(
            training_state,
            jax.local_devices()[:local_devices_to_use],
        )

        if not eval_env:
            eval_env = train_env
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn,
                rng=jax.random.split(eval_key, self.num_eval_envs),
            )

        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = wrap_for_training(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            randomization_fn=v_randomization_fn,
        )

        evaluator = MultiAgentEvaluator(
            eval_env,
            functools.partial(
                make_policies,
                deterministic=self.deterministic_eval,
            ),
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            key=eval_key,
        )

        # Run initial eval
        metrics = {}
        if process_id == 0 and config.num_evals > 1:
            metrics = evaluator.run_evaluation(
                _unpmap(
                    (
                        training_state.normalizer_params,
                        [ap.policy for ap in training_state.agent_params],
                    )
                ),
                training_metrics={},
            )
            progress_fn(
                0,
                metrics,
                make_policies,
                _unpmap(
                    (
                        training_state.normalizer_params,
                        [ap.policy for ap in training_state.agent_params],
                    )
                ),
                unwrapped_env,
            )

        training_metrics = {}
        training_walltime = 0
        current_step = 0
        for eval_epoch_num in range(num_evals_after_init):
            logging.info(
                "starting iteration %s %s",
                eval_epoch_num,
                time.time() - xt,
            )

            for _ in range(max(self.num_resets_per_eval, 1)):
                # optimization
                epoch_key, local_key = jax.random.split(local_key)
                epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
                (
                    training_state,
                    env_state,
                    training_metrics,
                ) = training_epoch_with_timing(
                    training_state,
                    env_state,
                    epoch_keys,
                )
                current_step = int(_unpmap(training_state.env_steps))

                key_envs = jax.vmap(
                    lambda x, s: jax.random.split(x[0], s),
                    in_axes=(0, None),
                )(key_envs, key_envs.shape[1])
                # TODO: move extra reset logic to the AutoResetWrapper.
                env_state = reset_fn(key_envs) if self.num_resets_per_eval > 0 else env_state

                if process_id == 0:
                    # Run evals.
                    metrics = evaluator.run_evaluation(
                        _unpmap(
                            (
                                training_state.normalizer_params,
                                [ap.policy for ap in training_state.agent_params],
                            )
                        ),
                        training_metrics,
                    )
                    do_render = (eval_epoch_num % config.visualization_interval) == 0
                    progress_fn(
                        current_step,
                        metrics,
                        make_policies,
                        _unpmap(
                            (
                                training_state.normalizer_params,
                                [ap.policy for ap in training_state.agent_params],
                            )
                        ),
                        unwrapped_env,
                        do_render=do_render,
                    )

        total_steps = current_step
        assert total_steps >= config.total_env_steps

        # If there was no mistakes the training_state should still be identical on all
        # devices.
        pmap.assert_is_replicated(training_state)
        params = _unpmap(
            (
                training_state.normalizer_params,
                [ap.policy for ap in training_state.agent_params],
            )
        )
        logging.info("total steps: %s", total_steps)
        pmap.synchronize_hosts()
        return make_policies, params, metrics

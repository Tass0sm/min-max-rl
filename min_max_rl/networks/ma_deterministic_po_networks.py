"""Multi-Agent Policy Optimization Networks."""

from typing import Sequence, Tuple

import jax
from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen

from . import my_networks


@flax.struct.dataclass
class DeterministicPiNetwork:
  policy_network: networks.FeedForwardNetwork


def make_inference_fn(network: DeterministicPiNetwork):
  """Creates params and inference function for the PPO agent."""

  def make_policy(
      params: types.Params
  ) -> types.Policy:
    policy_network = network.policy_network

    def policy(
        observations: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
      param_subset = (params[0], params[1])  # normalizer and policy params
      action = policy_network.apply(*param_subset, observations)
      return action, {}

    return policy

  return make_policy


def make_inference_fns(ma_deterministic_po_networks: list[DeterministicPiNetwork]):
    """Creates params and inference function for all agents."""

    # we can reuse make_inference_fn from ppo because it doesn't use the value
    # function.
    make_policy_fns = [make_inference_fn(po_network) for po_network in ma_deterministic_po_networks]

    def make_policies(
            params: tuple[types.PreprocessorParams, list[types.Params]], deterministic: bool = False
    ) -> types.Policy:
        preprocessor_params, agent_params = params
        policies = [make_policy_fn((preprocessor_params, params)) for make_policy_fn, params in zip(make_policy_fns, agent_params)]
        return policies

    return make_policies


def make_deterministic_pi_network(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = 'state',
) -> DeterministicPiNetwork:
  """Make Pi network with preprocessor."""
  policy_network = my_networks.make_policy_network(
      action_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      bias_init = jax.nn.initializers.constant(0.5),
      obs_key=policy_obs_key,
  )

  return DeterministicPiNetwork(
      policy_network=policy_network,
  )


def make_ma_deterministic_po_networks(
        n_agents: int,
        observation_size: types.ObservationSize,
        action_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
        activation: networks.ActivationFn = linen.swish,
        policy_obs_key: str = 'state',
) -> list[DeterministicPiNetwork]:
    """Make policy networks with preprocessor."""
  
    return [
        make_deterministic_pi_network(observation_size,
                                      action_size,
                                      preprocess_observations_fn,
                                      policy_hidden_layer_sizes,
                                      activation,
                                      policy_obs_key) for _ in range(n_agents)
    ]

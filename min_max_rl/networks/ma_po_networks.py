"""Multi-Agent Policy Optimization Networks."""

from typing import Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen

from brax.training.agents.ppo.networks import make_inference_fn


@flax.struct.dataclass
class PiNetwork:
  policy_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fns(ma_po_networks: list[PiNetwork]):
    """Creates params and inference function for all agents."""

    # we can reuse make_inference_fn from ppo because it doesn't use the value
    # function.
    make_policy_fns = [make_inference_fn(po_network) for po_network in ma_po_networks]

    def make_policies(
            params: tuple[types.PreprocessorParams, list[types.Params]], deterministic: bool = False
    ) -> types.Policy:
        preprocessor_params, agent_params = params
        policies = [make_policy_fn((preprocessor_params, params)) for make_policy_fn, params in zip(make_policy_fns, agent_params)]
        return policies

    return make_policies


def make_pi_network(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = 'state',
) -> PiNetwork:
  """Make Pi network with preprocessor."""
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size
  )
  policy_network = networks.make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      obs_key=policy_obs_key,
  )

  return PiNetwork(
      policy_network=policy_network,
      parametric_action_distribution=parametric_action_distribution,
  )


def make_ma_po_networks(
        n_agents: int,
        observation_size: types.ObservationSize,
        action_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
        activation: networks.ActivationFn = linen.swish,
        policy_obs_key: str = 'state',
) -> list[PiNetwork]:
    """Make policy networks with preprocessor."""
  
    return [
        make_pi_network(observation_size,
                        action_size,
                        preprocess_observations_fn,
                        policy_hidden_layer_sizes,
                        activation,
                        policy_obs_key) for _ in range(n_agents)
    ]

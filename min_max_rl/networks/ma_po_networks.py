"""Multi-Agent Policy Optimization Networks."""

from typing import Sequence, Tuple, Callable

import jax
import jax.numpy as jnp
from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen

from brax.training.agents.ppo.networks import make_inference_fn
from . import my_networks


@flax.struct.dataclass
class PiNetwork:
  policy_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution


class IdentityBijector:
  """Identity Bijector."""

  def forward(self, x):
    return x

  def inverse(self, y):
    return y

  def forward_log_det_jacobian(self, x):
    return 0


class NormalDistribution(distribution.ParametricDistribution):
  """Normal distribution"""

  def __init__(self, event_size):
    """Initialize the distribution.

    Args:
      event_size: the size of events (i.e. actions).
    """
    super().__init__(
        param_size=event_size,
        postprocessor=IdentityBijector(),
        event_ndims=1,
        reparametrizable=True,
    )

  def create_dist(self, parameters):
    loc = parameters
    scale = jnp.ones_like(loc) * 1.22140275816
    return distribution.NormalDistribution(loc=loc, scale=scale)


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


def make_normal_dist_network(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    activation: networks.ActivationFn = linen.swish,
    bias_init = jax.nn.initializers.constant(1.0),
    bias: bool = True,
    layer_norm: bool = False,
    policy_obs_key: str = 'state',
) -> PiNetwork:
  """Make Pi network with preprocessor."""
  parametric_action_distribution = NormalDistribution(
    event_size=action_size,
  )
  policy_network = my_networks.make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      bias_init=bias_init,
      bias=bias,
      layer_norm=layer_norm,
      obs_key=policy_obs_key,
  )

  return PiNetwork(
      policy_network=policy_network,
      parametric_action_distribution=parametric_action_distribution,
  )


def make_pi_network(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    activation: networks.ActivationFn = linen.swish,
    bias_init = jax.nn.initializers.constant(1.0),
    policy_obs_key: str = 'state',
) -> PiNetwork:
  """Make Pi network with preprocessor."""
  parametric_action_distribution = distribution.NormalTanhDistribution(
    event_size=action_size,
    var_scale=1.0,
  )
  policy_network = my_networks.make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      bias_init=bias_init,
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
        make_network_fn: Callable = make_pi_network,
        make_network_fns: list[Callable] = None,
        policy_obs_key: str = 'state',
) -> list[PiNetwork]:
    """Make policy networks with preprocessor."""

    if make_network_fns is None:
        make_network_fns = [make_network_fn for _ in range(n_agents)]

    return [
        fn(observation_size,
           action_size,
           preprocess_observations_fn=preprocess_observations_fn,
           policy_hidden_layer_sizes=policy_hidden_layer_sizes,
           activation=activation,
           policy_obs_key=policy_obs_key) for fn in make_network_fns
    ]

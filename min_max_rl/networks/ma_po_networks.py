"""Multi-Agent Policy Optimization Networks."""

from typing import Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen

from brax.training.agents.ppo.networks import PPONetworks, make_inference_fn, make_ppo_networks


def make_inference_fns(ma_po_networks: list[PPONetworks]):
    """Creates params and inference function for the PPO agent."""

    make_policy_fns = [make_inference_fn(po_network) for po_network in ma_po_networks]

    def make_policies(
            agent_params: list[types.Params], deterministic: bool = False
    ) -> types.Policy:
        policies = [make_policy_fn(params) for make_policy_fn, params in zip(make_policy_fns, agent_params)]
        return policies

    return make_policies


def make_ma_po_networks(
        n_agents: int,
        observation_size: types.ObservationSize,
        action_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
        value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
        activation: networks.ActivationFn = linen.swish,
        policy_obs_key: str = 'state',
        value_obs_key: str = 'state',
) -> list[PPONetworks]:
    """Make PPO networks with preprocessor."""
  
    return [
        make_ppo_networks(observation_size,
                          action_size,
                          preprocess_observations_fn,
                          policy_hidden_layer_sizes,
                          value_hidden_layer_sizes,
                          activation,
                          policy_obs_key,
                          value_obs_key) for _ in range(n_agents)
    ]

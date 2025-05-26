from typing import Callable, Dict, Optional, Tuple

from brax import base, envs
from brax.base import System
from brax.envs.base import Wrapper
from brax.envs.wrappers.training import (
    VmapWrapper,
    EpisodeWrapper,
    AutoResetWrapper,
    EvalMetrics,
    EvalWrapper,
    DomainRandomizationVmapWrapper
)

from min_max_rl.envs.wrappers.ma_training_wrappers import MAEpisodeWrapper


def wrap_for_training(
        env,
        episode_length: int = 1000,
        action_repeat: int = 1,
        randomization_fn: Optional[
            Callable[[System], Tuple[System, System]]
        ] = None,
) -> Wrapper:
    assert isinstance(env, envs.Env), "env must be an instance of brax.envs.Env"

    if randomization_fn is None:
        env = VmapWrapper(env)
    else:
        env = DomainRandomizationVmapWrapper(env, randomization_fn)
    env = MAEpisodeWrapper(env, episode_length, action_repeat)
    env = AutoResetWrapper(env)
    return env

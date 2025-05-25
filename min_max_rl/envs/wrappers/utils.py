from brax import base, envs
from brax.v1 import envs as envs_v1


def wrap_for_training(train_env, **kwargs):
    if isinstance(train_env, envs.Env):
        return envs.training.wrap(train_env, **kwargs)
    else:
        return envs_v1.wrappers.wrap_for_training(train_env, **kwargs)

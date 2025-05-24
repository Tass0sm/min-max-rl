from min_max_rl.envs import custom_envs


def get_env(env_name: str, extra_envs: dict = {}, **kwargs):
    if hasattr(custom_envs, env_name):
        EnvClass = getattr(custom_envs, env_name)
    elif env_name in extra_envs:
        EnvClass = extra_envs[env_name]
    else:
        raise Exception(f"{env_name} not found")

    return EnvClass(**kwargs)

import mlflow

from min_max_rl.envs import get_env
from min_max_rl.utils import RunConfig
from min_max_rl.agents.cgd_po import CGD_PO


def progress_fn(num_steps, metrics, *args, **kwargs):
    print(f"Logging for {num_steps}")
    print(metrics)
    mlflow.log_metrics(metrics, step=num_steps)


def training_run(run_id, env, seed, agent_class, progress_fn=progress_fn, alg_hps={}, run_params={}, extras={}):
    agent = agent_class(**alg_hps)

    alg_hps = {
        **alg_hps,
        "seed": seed,
    }

    mlflow.log_params(alg_hps)

    run_config = RunConfig(
        env=env,
        seed=seed,
        **run_params
    )

    make_inference_fn, params, _ = agent.train_fn(
        run_config,
        env,
        progress_fn=progress_fn,
        **extras
    )

    with mlflow.MlflowClient()._log_artifact_helper(run_id, 'policy_params') as tmp_path:
        model.save_params(tmp_path, params)

    return make_inference_fn, params


def train_for_all(envs, func, alg_tag, seed_range=(0, 3), extra_envs={}, **kwargs):
    for env_name in envs:
        env = get_env(env_name, extra_envs=extra_envs)
        env_tag = type(env).__name__

        for seed in range(*seed_range):
            with mlflow.start_run(tags={"env": env_tag, "alg": alg_tag}) as run:
                func(run, env, seed, **kwargs)


def cgd_po_train(run, env, seed, run_params: dict = {}):
    return training_run(
        run.info.run_id,
        env,
        seed,
        agent_class=CGD_PO,
        alg_hps=env.cgd_po_hps,
        run_params=run_params,
        progress_fn=progress_fn,
        extras={}
    )


def main():
    max_seed = 1
    train_for_all(["LQGame"], cgd_po_train, "CGD_PO", seed_range=(0, max_seed), run_params = {
        # "total_env_steps": None,
        # "episode_length": None,
        # "num_envs": None,
        # "num_eval_envs": None,
        # "num_evals": None,
        # "action_repeat": None,
        # "max_devices_per_host": None,
    })


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")
    mlflow.set_experiment("proj2-final-experiments")

    main()

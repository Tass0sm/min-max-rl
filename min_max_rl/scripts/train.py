import mlflow

from min_max_rl.envs import get_env
from min_max_rl.agents.cgd_po import CGD_PO


def progress_fn(num_steps, metrics, *args, **kwargs):
    print(f"Logging for {num_steps}")
    print(metrics)
    mlflow.log_metrics(metrics, step=num_steps)


def training_run(run_id, env, seed, train_fn, progress_fn=progress_fn, hyperparameters={}, extras={}):
    hyperparameters = {
        **hyperparameters,
        "seed": seed,
    }

    mlflow.log_params(hyperparameters)

    train_fn = functools.partial(train_fn, **hyperparameters)

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress_fn,
        seed=seed,
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


def cgd_po_train(run, env, seed):
    pass


def main():
    max_seed = 1
    train_for_all(["LQGame"], cgd_po_train, "CGD_PO", seed_range=(0, max_seed))


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")
    mlflow.set_experiment("proj2-final-experiments")

    main()

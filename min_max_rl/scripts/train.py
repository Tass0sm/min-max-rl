import mlflow

from brax.io import model

from min_max_rl.envs import get_env
from min_max_rl.utils import RunConfig
from min_max_rl.agents.cvpgd import CVPGD
from min_max_rl.agents.cvpgo import CVPGO
from min_max_rl.agents.vpgda import VPGDA

from min_max_rl.agents.dpgda import DPGDA
from min_max_rl.agents.cdpgd import CDPGD


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


def cvpgo_train(run, env, seed, run_params: dict = {}):
    return training_run(
        run.info.run_id,
        env,
        seed,
        agent_class=CVPGO,
        alg_hps=env.cvpgd_hps,
        run_params=run_params,
        progress_fn=progress_fn,
        extras={}
    )


def cvpgd_train(run, env, seed, run_params: dict = {}):
    return training_run(
        run.info.run_id,
        env,
        seed,
        agent_class=CVPGD,
        alg_hps=env.cvpgd_hps,
        run_params=run_params,
        progress_fn=progress_fn,
        extras={}
    )


def vpgda_train(run, env, seed, run_params: dict = {}):
    return training_run(
        run.info.run_id,
        env,
        seed,
        agent_class=VPGDA,
        alg_hps=env.vpgda_hps,
        run_params=run_params,
        progress_fn=progress_fn,
        extras={}
    )


def dpgda_train(run, env, seed, run_params: dict = {}):
    return training_run(
        run.info.run_id,
        env,
        seed,
        agent_class=DPGDA,
        alg_hps=env.dpgda_hps,
        run_params=run_params,
        progress_fn=progress_fn,
        extras={}
    )

def cdpgd_train(run, env, seed, run_params: dict = {}):
    return training_run(
        run.info.run_id,
        env,
        seed,
        agent_class=CDPGD,
        alg_hps=env.cdpgd_hps,
        run_params=run_params,
        progress_fn=progress_fn,
        extras={}
    )


def main():
    max_seed = 1

    # train_for_all(["LQGame"], gda_po_train, "GDA_PO", seed_range=(0, max_seed), run_params = {
    #     "total_env_steps": 5_000_000,
    #     # "episode_length": None,
    #     "num_envs": 100,
    #     # "num_eval_envs": None,
    #     "num_evals": 10,
    #     # "action_repeat": None,
    #     # "max_devices_per_host": None,
    # })


    # train_for_all(["Bilinear"], vpgda_train, "VPGDA", seed_range=(0, max_seed), run_params = {
    #     "total_env_steps": 1_000_000,
    #     "episode_length": 1,
    #     "num_envs": 128,
    #     "num_eval_envs": 128,
    #     "num_evals": 50,
    #     # "action_repeat": None,
    #     # "max_devices_per_host": None,
    # })

    train_for_all(["Bilinear"], cdpgd_train, "CDPGD", seed_range=(0, max_seed), run_params = {
        "total_env_steps": 1_000_000,
        "episode_length": 1,
        "num_envs": 128,
        "num_eval_envs": 128,
        "num_evals": 50,
        # "action_repeat": None,
        # "max_devices_per_host": None,
    })

    # train_for_all(["Bilinear"], dpgda_train, "DPGDA", seed_range=(0, max_seed), run_params = {
    #     "total_env_steps": 1_000_000,
    #     "episode_length": 1,
    #     "num_envs": 1,
    #     "num_eval_envs": 1,
    #     "num_evals": 50,
    #     # "action_repeat": None,
    #     # "max_devices_per_host": None,
    # })

    # train_for_all(["Bilinear"], gda_po_train, "GDA_PO", seed_range=(0, max_seed), run_params = {
    #     "total_env_steps": 5_000_000,
    #     "episode_length": 1,
    #     "num_envs": 256,
    #     "num_eval_envs": 256,
    #     "num_evals": 1000,
    #     # "action_repeat": None,
    #     # "max_devices_per_host": None,
    # })


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")
    mlflow.set_experiment("abhijeet-experiments")

    main()

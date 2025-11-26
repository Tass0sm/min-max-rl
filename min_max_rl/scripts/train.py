import argparse
import tempfile
from typing import Optional
from pathlib import Path

import mlflow

from brax.io import model

from min_max_rl import agents
from min_max_rl.envs import get_env
from min_max_rl.utils import RunConfig


def main():
    parser = argparse.ArgumentParser(description="General training script for agents.")
    parser.add_argument('--agent', type=str, help='Name of agent for training.')
    parser.add_argument('--env', type=str, help='Name of environment to consider for training.')

    # run config arguments
    parser.add_argument('--total_env_steps', type=int, default=1000000, help='Number of timesteps allowed for training.')
    parser.add_argument('--num_envs', type=int, default=1, help='Number of environments across which to vectorize.')
    parser.add_argument('--num_evals', type=int, default=16, help='Number of evaluations to perform and log during training.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for randomness in training.')
    args = parser.parse_args()

    if hasattr(agents, args.agent):
        AgentClass = getattr(agents, args.agent)
    else:
        raise NotImplementedError(f"Agent {args.agent} not found")

    def progress_fn(current_step, metrics, *args, **kwargs):
        print(f"Logging for {current_step}")
        print(metrics)
        mlflow.log_metrics(metrics, step=current_step)

        # params = kwargs["params"]

        # # save params to temporary directory and then log the saved file to store in mlflow
        # with tempfile.TemporaryDirectory() as tmp_dir:
        #     path = Path(tmp_dir, f"policy_params_{current_step}")
        #     model.save_params(path, params)
        #     mlflow.log_artifact(path)

    env = get_env(args.env)

    agent_hps = getattr(env, f"{args.agent.lower()}_hps")

    agent = AgentClass(**agent_hps)

    run_config = RunConfig(
        seed=args.seed,
        total_env_steps=args.total_env_steps,
        num_envs=args.num_envs,
        num_evals=args.num_evals,
        # **run_params
    )

    with mlflow.start_run(tags={"env": args.env, "agent": args.agent}) as run:

        mlflow.log_params(agent_hps)

        make_policy, params, metrics = agent.train_fn(
            config=run_config,
            train_env=env,
            progress_fn=progress_fn
        )

        # # save params to temporary directory and then log the saved file to store in mlflow
        # with tempfile.TemporaryDirectory() as tmp_dir:
        #     path = Path(tmp_dir, f"policy_params")
        #     model.save_params(path, params)
        #     mlflow.log_artifact(path)


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")
    mlflow.set_experiment("abhijeet-experiments")

    main()

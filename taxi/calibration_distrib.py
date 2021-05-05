from concurrent.futures import ProcessPoolExecutor
from urllib.parse import quote_plus

import gym
import optuna

from taxi.agent_max import Agent
from taxi.monitor import interact

STUDY_NAME = "taxi_distrib_max"
STORAGE = f"mysql+mysqlconnector://optuna:{quote_plus('optun@')}@localhost/optuna"
NUM_WORKERS = 6


def objective(trial: optuna.Trial):
    env = gym.make("Taxi-v3")

    alpha = trial.suggest_uniform("alpha", low=0.0001, high=0.7)
    epsilon = trial.suggest_uniform("epsilon", low=0.0001, high=0.5)
    gamma = trial.suggest_uniform("gamma", low=0.5, high=1.0)
    epsilon_decay = trial.suggest_loguniform("epsilon_decay", low=1.0, high=5.0)

    agent = Agent(epsilon, alpha, gamma, epsilon_decay)

    avg_reward, best_avg_reward = interact(env, agent, log=False)
    return best_avg_reward


def main():
    print("Created study for sub-process")
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        storage=STORAGE,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=20)
    print("Optimization finished for sub-process")


if __name__ == "__main__":
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        storage=STORAGE,
        load_if_exists=True,
    )

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        for _ in range(NUM_WORKERS):
            pool.submit(main)

    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    print(study.best_params)

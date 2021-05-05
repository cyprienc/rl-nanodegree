from concurrent.futures import ThreadPoolExecutor

import gym
import optuna

from taxi.agent import Agent
from taxi.monitor import interact


def objective(trial: optuna.Trial):
    env = gym.make("Taxi-v3")
    agent = Agent()

    agent.alpha = trial.suggest_uniform("alpha", low=0.0001, high=0.9999)
    agent.epsilon = trial.suggest_uniform("epsilon", low=0.0001, high=0.9999)
    agent.gamma = trial.suggest_uniform("gamma", low=0.0001, high=1.0)
    agent.epsilon_decay = trial.suggest_loguniform("epsilon_decay", low=1.0, high=20.0)
    avg_reward, best_avg_reward = interact(env, agent, log=False)
    return best_avg_reward


if __name__ == "__main__":
    study = optuna.create_study(study_name="taxi", direction="maximize")
    with ThreadPoolExecutor(max_workers=5) as executor:
        for _ in range(5):
            executor.submit(study.optimize, objective, 20)
    print(study.best_params)

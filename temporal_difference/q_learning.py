import sys
from collections import defaultdict
from typing import Dict

import gym
import numpy as np

from temporal_difference.utils import epsilon_greedy_action


def q_learning(
    env: gym.Env,
    num_episodes: int,
    alpha: float,
    gamma: float = 1.0,
) -> Dict[int, np.array]:
    # initialize action-value function (empty dictionary of arrays)
    nA = env.nA
    Q = defaultdict(lambda: np.zeros(nA))
    # initialize performance monitor
    # loop over episodes
    epsilon = 0.1
    for i_episode in range(1, num_episodes + 1):
        if (i_episode % 100) == 0:
            epsilon /= 2.0
        # monitor progress
        if i_episode % 100 == 0:
            print(
                "\rEpisode {}/{} - Epsilon {}".format(i_episode, num_episodes, epsilon),
                end="",
            )
            sys.stdout.flush()

            obs = env.reset()

            done = False
            while not done:
                action = epsilon_greedy_action(Q, obs, nA, epsilon)
                new_obs, reward, done, info = env.step(action)
                Q[obs][action] += alpha * (
                    reward + gamma * np.max(Q[new_obs]) - Q[obs][action]
                )
                obs = new_obs

    return Q

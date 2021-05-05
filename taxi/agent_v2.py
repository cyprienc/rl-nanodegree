from collections import defaultdict
from warnings import catch_warnings

import numpy as np

from taxi.utils import epsilon_greedy_action, greedy_action


class Agent:
    def __init__(self, nA=6):
        """Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.N = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.1
        self.n_episode = 1
        self.alpha = 0.5
        self.gamma = 1.0
        self.c = 10
        self.n_state = defaultdict(lambda: 1)

    def select_action(self, state):
        """Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        if np.random.choice([0, 1], size=1, p=[self.epsilon, 1 - self.epsilon]).item():
            # Greedy step
            action = greedy_action(self.Q, state)
        else:
            # Exploration step
            t = self.n_state[state]
            with np.errstate(divide="ignore"):
                uncertainty = self.c * np.sqrt(np.log(t) / self.N[state])
                uncertainty = np.nan_to_num(uncertainty, nan=1e6)
            ucb_values = self.Q[state] + uncertainty
            action = np.random.choice(np.flatnonzero(ucb_values == ucb_values.max()))

        self.n_state[state] += 1
        self.N[state][action] += 1
        return action

    def step(self, state, action, reward, next_state, done):
        """Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # p_vec = np.full(self.Q[next_state].shape, self.epsilon / (self.nA - 1))
        # max_action = self.Q[next_state].argmax()
        # p_vec[max_action] = 1 - self.epsilon
        #
        # self.Q[state][action] += self.alpha * (
        #     reward
        #     + self.gamma * (np.dot(p_vec, self.Q[next_state]))
        #     - self.Q[state][action]
        # )

        # Q-Learning
        self.Q[state][action] += self.alpha * (
            reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action]
        )

        if done:
            self.n_episode += 1
            if (self.n_episode % 1000) == 0:
                self.epsilon /= 2.0

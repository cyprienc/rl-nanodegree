from collections import defaultdict

import numpy as np

from taxi.utils import epsilon_greedy_action


class Agent:
    def __init__(self, epsilon, alpha, gamma, epsilon_decay, nA=6):
        """Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay

        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.n_episode = 1

    def select_action(self, state):
        """Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return epsilon_greedy_action(self.Q, state, self.nA, self.epsilon)

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
        if not done:
            p_vec = np.full(self.Q[next_state].shape, self.epsilon / (self.nA - 1))
            max_action = self.Q[next_state].argmax()
            p_vec[max_action] = 1 - self.epsilon

            self.Q[state][action] += self.alpha * (
                reward
                + self.gamma * (np.dot(p_vec, self.Q[next_state]))
                - self.Q[state][action]
            )
        else:
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])

            self.n_episode += 1
            if (self.n_episode % 100) == 0:
                self.epsilon /= self.epsilon_decay

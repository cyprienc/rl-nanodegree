import numpy as np


def greedy_action(Q, state):
    Q_values = Q[state]
    return np.random.choice(np.flatnonzero(Q_values == Q_values.max()))


def epsilon_greedy_action(Q, state, nA: int, epsilon: float) -> int:

    if np.random.choice([0, 1], size=1, p=[epsilon, 1 - epsilon]).item():
        # Greedy step
        return greedy_action(Q, state)
    else:
        # Exploration step
        return np.random.randint(0, nA)

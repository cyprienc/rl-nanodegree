import gym
import numpy as np

import check_test
from temporal_difference.sarsa import sarsa

env = gym.make("CliffWalking-v0")

print(env.action_space)
print(env.observation_space)

V_opt = np.zeros((4, 12))
V_opt[0:13][0] = -np.arange(3, 15)[::-1]
V_opt[0:13][1] = -np.arange(3, 15)[::-1] + 1
V_opt[0:13][2] = -np.arange(3, 15)[::-1] + 2
V_opt[3][0] = -13

print(V_opt)

# obtain the estimated optimal policy and corresponding action-value function
Q_sarsa = sarsa(env, 20_000, 0.5)

# print the estimated optimal policy
policy_sarsa = np.array(
    [np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]
).reshape(4, 12)
check_test.run_check("td_control_check", policy_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsa)

# plot the estimated optimal state-value function
print("\nEstimated optimal state-value function:")
V_sarsa = np.array([np.round(np.max(Q_sarsa[key]), decimals=1) if key in Q_sarsa else 0 for key in np.arange(48)]).reshape(4, 12)
print(V_sarsa)

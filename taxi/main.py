import gym
from monitor import interact

from agent_max import Agent

env = gym.make("Taxi-v3")

epsilon = 0.351
alpha = 0.259
epsilon_decay = 3.03
gamma = 0.742

agent = Agent(epsilon, alpha, gamma, epsilon_decay)
avg_rewards, best_avg_reward = interact(env, agent)

from collections import deque

import numpy as np
import torch


def run(
    env,
    agent,
    env_solve_criteria=13.0,
    n_episodes=2000,
    max_t=1000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.99,
    train_mode=False,
):
    """Run the specified Agent in the given env

    Params
    ======
        env: Unity ML Env
        agent: Agent
        env_solve_criteria (float): minimum average score over 100 episodes to consider the env solved
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        train_mode (bool): flag for train mode
    """
    brain_name = env.brain_names[0]

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=train_mode)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]

        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            try:
                env_info = env.step(action)[brain_name]
            except Exception as e:
                print(e)

            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]

            if train_mode:
                agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        mean_score_window = np.mean(scores_window)
        print(
            f"\rEpisode {i_episode}\tAverage Score: {mean_score_window:.2f}",
            end="",
        )
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {mean_score_window:.2f}")
        if train_mode and mean_score_window >= env_solve_criteria:
            print(
                f"\nEnvironment solved in {i_episode:d} episodes!\tAverage Score: {mean_score_window:.2f}"
            )
            torch.save(agent.qnetwork_local.state_dict(), "model.pth")
            break
    return scores

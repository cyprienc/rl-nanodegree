from collections import deque

import numpy as np
import torch

from agent import MultiAgentActorCritic


def run(
    env,
    agent: MultiAgentActorCritic,
    env_solve_criteria: float = 0.5,
    n_episodes: int = 2000,
    max_t: int = 10_000,
    train_mode: bool = False,
):
    """Run the specified Agent in the given env

    Params
    ======
        env: Unity ML Env
        agent: Agent
        env_solve_criteria (float): minimum average score over 100 episodes to consider the env solved
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        train_mode (bool): flag for train mode
    """
    brain_name = env.brain_names[0]

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1, n_episodes + 1):

        env_info = env.reset(train_mode=train_mode)[brain_name]  # reset the environment

        states = env_info.vector_observations

        score = np.zeros_like(env_info.rewards)
        for t in range(max_t):
            actions = agent.act(states, train_mode=train_mode)
            try:
                env_info = env.step(actions)[brain_name]
            except Exception as e:
                print(e)
                raise

            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done

            if train_mode:
                agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += np.array(rewards)
            if np.any(dones):
                break

        score = np.mean(score)
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        mean_score_window = np.mean(scores_window)
        print(
            f"\rEpisode {i_episode}\tScore:{score:.3f}\tAverage Score: {mean_score_window:.3f}",
            end="",
        )
        if i_episode % 5 == 0:
            print(
                f"\rEpisode {i_episode}\tScore:{score:.3f}\tAverage Score: {mean_score_window:.3f}"
            )
        if train_mode and mean_score_window >= env_solve_criteria:
            print(
                f"\nEnvironment solved in {i_episode:d} episodes!\tAverage Score: {mean_score_window:.3f}"
            )
            for i, sub_agent in enumerate(agent.agents):
                torch.save(sub_agent.state_dict(), f"agent_{i}.pth")
            break
    return scores


def random_play(env, agent: MultiAgentActorCritic, n_episodes: int, max_t: int = 1000):
    """
    Plays random episode in the passed env and store the transitions in the agent's buffer
    Args:
        env: Unity ML Env
        agent: Agent
        n_episodes: number of random episodes
        max_t: maximum number of time steps per episode

    """
    brain_name = env.brain_names[0]

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment

        states = env_info.vector_observations

        score = np.zeros_like(env_info.rewards)
        for t in range(max_t):
            actions = np.clip(
                np.random.randn(agent.num_agents, agent.action_space_size), -1, 1
            )
            try:
                env_info = env.step(actions)[brain_name]
            except Exception as e:
                print(e)
                raise

            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done

            agent.step(states, actions, rewards, next_states, dones, learn=False)
            states = next_states
            score += np.array(rewards)

            if np.any(dones):
                break

        print(
            f"\rEpisode {i_episode}\tScore:{np.mean(score):.3f}",
            end="",
        )

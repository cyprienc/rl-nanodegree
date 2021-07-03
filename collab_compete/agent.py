import random
from collections import deque, namedtuple
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import ActorNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiAgentActorCritic:
    def __init__(
        self,
        num_agents: int,
        state_size: int,
        action_space_size: int,
        buffer_size: int = int(1e6),
        batch_size: int = 128,
        seed: int = 42,
        gamma: float = 0.95,
        critic_lr: float = 1e-3,
        actor_lr: float = 1e-3,
        tau: float = 5e-3,
        sigma: float = 0.2,
        sigma_scale: float = 0.9999,
        min_sigma: float = 0.01,
        reward_scale: float = 10.0,
        update_every: int = 1,
        update_policy_every: int = 4,
        policy_smoothing: float = 0.2,
        clip: float = 0.5,
    ):
        """
        Actor-Critic multi-agent trainer for collaborative environment

        Based of MADDPG, with:
            - Clipped Double-Q Learning (TD3)
            - “Delayed” Policy Updates (TD3)
            - Target Policy Smoothing (TD3)
            - Centralized Critic (FACMAC)


        Args:
            num_agents: number of agents
            state_size: observation space size for a single agent
            action_space_size: action space size for a single agent
            buffer_size: size of the replay buffer
            batch_size: batch size for the SGD
            seed: random seed for reproducibility
            gamma: discount factor
            critic_lr: learning rate for the critic networks
            actor_lr: learning rate for the actor networks
            tau: soft-update weight
            sigma: initial noise for exploration
            sigma_scale: noise weight decay per step
            min_sigma: minimum value of sigma
            reward_scale: reward weight
            update_every: critic update frequency
            update_policy_every: policy update frequency (per critic update)
            policy_smoothing: noise for policy smoothing during critic update
            clip: noise clipping
        """
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        self.num_agents = num_agents
        self.state_size = state_size
        self.action_space_size = action_space_size

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.tau = tau
        self.sigma = sigma
        self.sigma_scale = sigma_scale
        self.min_sigma = min_sigma
        self.reward_scale = reward_scale
        self.update_every = update_every
        self.update_policy_every = update_policy_every
        self.policy_smoothing = policy_smoothing
        self.clip = clip
        self.t_step = 0
        self.p_step = 0

        # Base Agent
        self.agents = [
            ActorNetwork(
                agent_obs_space_size=state_size,
                agent_action_space_size=action_space_size,
                total_obs_space_size=state_size * num_agents,
                total_action_space_size=action_space_size * num_agents,
            ).to(device)
            for _ in range(num_agents)
        ]
        # Base Agent's twin
        self.agents_twin = [
            ActorNetwork(
                agent_obs_space_size=state_size,
                agent_action_space_size=action_space_size,
                total_obs_space_size=state_size * num_agents,
                total_action_space_size=action_space_size * num_agents,
            ).to(device)
            for _ in range(num_agents)
        ]

        # Target Agent
        self.target_agents = [
            ActorNetwork(
                agent_obs_space_size=state_size,
                agent_action_space_size=action_space_size,
                total_obs_space_size=state_size * num_agents,
                total_action_space_size=action_space_size * num_agents,
            ).to(device)
            for _ in range(num_agents)
        ]
        # Target agent's twin
        self.target_agents_twin = [
            ActorNetwork(
                agent_obs_space_size=state_size,
                agent_action_space_size=action_space_size,
                total_obs_space_size=state_size * num_agents,
                total_action_space_size=action_space_size * num_agents,
            ).to(device)
            for _ in range(num_agents)
        ]

        self.actor_parameters = []
        self.target_actor_parameters = []
        self.critic_parameters = []
        self.target_critic_parameters = []
        for agent, twin, target_agent, target_agent_twin in zip(
            self.agents, self.agents_twin, self.target_agents, self.target_agents_twin
        ):
            target_agent.load_state_dict(agent.state_dict())
            target_agent_twin.load_state_dict(twin.state_dict())

            # Notice that the twin's actor network is not used
            self.actor_parameters.extend(agent.actor_params)
            self.target_actor_parameters.extend(target_agent.actor_params)

            self.critic_parameters.extend(agent.critic_params)
            self.critic_parameters.extend(twin.critic_params)

            self.target_critic_parameters.extend(target_agent.critic_params)
            self.target_critic_parameters.extend(target_agent_twin.critic_params)

        self.actor_optimizer = optim.Adam(self.actor_parameters, lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_parameters, lr=self.critic_lr)

        self.memory = ReplayBuffer(
            buffer_size=self.buffer_size, batch_size=self.batch_size, seed=seed
        )

    def step(
        self,
        states: np.array,
        actions: np.array,
        rewards: List[float],
        next_states: np.array,
        dones: List[float],
        learn: bool = True,
    ) -> None:
        """
        Stores the transition in memory and possibly update the agent

        Args:
            states: states (n_agent, obs_space)
            actions: actions (n_agent, act_space)
            rewards: rewards (n_agent, )
            next_states: next states (n_agent, obs_space)
            dones: terminal state flag (n_agent,)
            learn: boolean flag to disable learning (for random exploration)

        """
        self.memory.add(
            states,
            actions,
            self.reward_scale * np.array(rewards),
            next_states,
            np.array(dones),
        )

        if learn:
            self.t_step = (self.t_step + 1) % self.update_every
            if self.t_step == 0:
                if len(self.memory) >= self.batch_size:
                    experiences = self.memory.sample()
                    self.learn(experiences)

    def act(self, states: np.array, train_mode: bool = False) -> np.array:
        """
        Take action for the given states

        Args:
            states: current states (n_agents, obs_size)
            train_mode: boolean to add exploration noise

        Returns: actions (n_agents, act_size)

        """
        states = (
            torch.from_numpy(state.reshape((1, -1))).float().to(device)
            for state in states
        )

        for agent in self.agents:
            agent.eval()

        with torch.no_grad():
            actions = [
                agent.actor(state).cpu().data.numpy()
                for agent, state in zip(self.agents, states)
            ]

        for agent in self.agents:
            agent.train()

        actions = np.vstack(actions)

        if train_mode:
            # Adding gaussian noise to the action for exploration
            # and clipping to ensure values are inside the action space's bounds
            actions = np.clip(
                actions + np.random.normal(loc=0, scale=self.sigma, size=actions.shape),
                a_min=-1,
                a_max=1,
            )
            # Weight decay of the noise
            self.sigma = max(self.min_sigma, self.sigma_scale * self.sigma)

        return actions

    def learn(self, experiences) -> None:
        """
        Performs an update of the critics and possible the policy

        Args:
            experiences: namedtuple sampled from the memory buffer

        """
        states, actions, rewards, next_states, dones = experiences

        # Critic update
        with torch.no_grad():
            # total reward for the centralized critic
            total_reward = rewards.sum(dim=1, keepdim=True)
            done, _ = dones.max(dim=1, keepdim=True)

            # computing next actions using target actors
            next_actions = torch.cat(
                [
                    a.actor(next_states[:, j, :])
                    for j, a in enumerate(self.target_agents)
                ],
                dim=1,
            )

            # policy smoothing
            next_actions = torch.clamp(
                next_actions
                + torch.clamp(
                    self.policy_smoothing * torch.randn_like(next_actions),
                    min=-self.clip,
                    max=self.clip,
                ),
                min=-1,
                max=1,
            )

            # very simple factoring of the target critics
            # in FACMAC, critics are factored with a non-linear monotonic function
            # here, we use the sum to factor them
            next_target_q = torch.sum(
                torch.cat(
                    [
                        a.critic(next_states.view((self.batch_size, -1)), next_actions)
                        for a in self.target_agents
                    ],
                    dim=1,
                ),
                dim=1,
                keepdim=True,
            )

            # factoring the Q-values for the twin target critics
            next_target_q_twin = torch.sum(
                torch.cat(
                    [
                        a.critic(next_states.view((self.batch_size, -1)), next_actions)
                        for a in self.target_agents_twin
                    ],
                    dim=1,
                ),
                dim=1,
                keepdim=True,
            )

            # minimizing Q-values are taken to reduce overestimation bias
            y = total_reward + (1 - done) * self.gamma * torch.min(
                next_target_q, next_target_q_twin
            )

        # factoring the Q-values for the base critic
        prediction = torch.sum(
            torch.cat(
                [
                    a.critic(
                        states.view((self.batch_size, -1)),
                        actions.view((self.batch_size, -1)),
                    )
                    for a in self.agents
                ],
                dim=1,
            ),
            dim=1,
            keepdim=True,
        )

        # factoring the Q-values for the twin critic
        prediction_twin = torch.sum(
            torch.cat(
                [
                    a.critic(
                        states.view((self.batch_size, -1)),
                        actions.view((self.batch_size, -1)),
                    )
                    for a in self.agents_twin
                ],
                dim=1,
            ),
            dim=1,
            keepdim=True,
        )

        loss = F.mse_loss(prediction, y) + F.mse_loss(prediction_twin, y)

        self.critic_optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic_parameters, 1.0)

        self.critic_optimizer.step()

        self.p_step = (self.p_step + 1) % self.update_policy_every

        if self.p_step == 0:
            # Policy update

            # new actions under the current policy
            actions = torch.cat(
                [a.actor(states[:, i, :]) for i, a in enumerate(self.agents)], dim=1
            )

            # Factored Q-values of the base critic for the current policy
            q = torch.sum(
                torch.cat(
                    [
                        a.critic(states.view((self.batch_size, -1)), actions)
                        for a in self.agents
                    ],
                    dim=1,
                ),
                dim=1,
                keepdim=True,
            )

            policy_loss = -q.mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_parameters, 1.0)
            self.actor_optimizer.step()

            # ----- update target network ----- #

            self._soft_update(self.target_critic_parameters, self.critic_parameters)
            self._soft_update(self.target_actor_parameters, self.actor_parameters)

    def _soft_update(self, target_params, local_params) -> None:
        for target_param, local_param in zip(target_params, local_params):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["states", "actions", "rewards", "next_states", "dones"],
        )
        random.seed(seed)

    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.stack([e.states for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.stack([e.actions for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.stack([e.rewards for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.stack([e.next_states for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.stack([e.dones for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

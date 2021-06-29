import random
from collections import deque, namedtuple
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import MultiAgentActorCriticNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiAgentActorCritic:
    def __init__(
        self,
        num_agents: int,
        num_envs: int,
        state_size: int,
        action_space_size: int,
        adversary_agents_indices: List[int],
        buffer_size: int = int(1e6),
        batch_size: int = 64,
        seed: int = 42,
        update_per_ts: int = 2,
        gamma: float = 0.99,
        critic_lr: float = 1e-3,
        actor_lr: float = 1e-3,
        tau: float = 1e-3,
    ):
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        self.num_agents = num_agents
        self.num_envs = num_envs
        self.state_size = state_size
        self.action_space_size = action_space_size

        self.adversary_agents_indices = adversary_agents_indices
        self.cooperative_agents_indices = [i for i in range(num_agents) if i not in adversary_agents_indices]

        self.batch_size = batch_size
        self.update_per_ts = update_per_ts
        self.gamma = gamma
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.tau = tau

        # Centralized action-value functions
        self.actor_critic_network = MultiAgentActorCriticNetwork(
            num_agents=num_agents,
            state_size=state_size,
            action_space_size=action_space_size,
        )

        self.actor_critic_target_network = MultiAgentActorCriticNetwork(
            num_agents=num_agents,
            state_size=state_size,
            action_space_size=action_space_size,
        )
        self.actor_critic_target_network.load_state_dict(
            self.actor_critic_network.state_dict()
        )

        self.critic_loss = nn.MSELoss()

        self.critic_optimizer = optim.Adam(
            self.actor_critic_network.critic_parameters,
            lr=self.critic_lr,
        )

        self.actor_optimizer = optim.Adam(
            self.actor_critic_network.actor_parameters,
            lr=self.actor_lr,
        )

        self.memory = ReplayBuffer(
            buffer_size=buffer_size, batch_size=batch_size, seed=seed
        )

    def step(self, states, actions, rewards, next_states, dones):
        states = states.reshape((self.num_envs, self.num_agents, -1))
        actions = actions.reshape((self.num_envs, self.num_agents, -1))
        rewards = rewards.reshape((self.num_envs, self.num_agents, -1))
        next_states = next_states.reshape((self.num_envs, self.num_agents, -1))
        dones = dones.reshape((self.num_envs, self.num_agents, -1))
        self.memory.add(states, actions, rewards, next_states, dones)

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.batch_size:
            for i in range(self.update_per_ts):
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, states):
        states = (
            torch.from_numpy(states.reshape((self.num_envs, self.num_agents, -1)))
            .float()
            .to(device)
        )

        self.actor_critic_network.eval()
        with torch.no_grad():
            logits = self.actor_critic_network.actors(states)
            cat = torch.distributions.categorical.Categorical(logits=logits)
            actions = cat.sample()
        self.actor_critic_network.train()

        return actions.cpu().data.numpy().flatten()

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # ---- Critic loss ---- #

        with torch.no_grad():
            logits = self.actor_critic_target_network.actors(next_states)
            cat = torch.distributions.categorical.Categorical(logits=logits)
            next_actions = cat.sample()
            y = rewards + self.gamma * (
                1 - dones
            ) * self.actor_critic_target_network.critic(next_states, next_actions)

        predictions = self.actor_critic_network.critic(states, actions)
        critic_loss = self.critic_loss(predictions, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.actor_critic_network.critic_parameters, 1)

        self.critic_optimizer.step()

        # ----- Policy loss ----- #

        logits = self.actor_critic_network.actors(states)
        cat = torch.distributions.categorical.Categorical(logits=logits)
        sample = cat.sample()

        policy_loss = -(
            cat.log_prob(sample)
            * torch.squeeze(self.actor_critic_network.critic(states, sample))
        ).mean()

        # Gradient clipping
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic_network.actor_parameters, 1)
        self.actor_optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update()

    def soft_update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            self.actor_critic_target_network.parameters(),
            self.actor_critic_network.parameters(),
        ):
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
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        random.seed(seed)
        self.seed = seed

    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        for state, action, reward, next_state, done in zip(
            states, actions, rewards, next_states, dones
        ):
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.stack([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.squeeze(np.stack([e.action for e in experiences if e is not None]))
            )
            .float()
            .to(device)
        )
        rewards = (
            torch.from_numpy(np.stack([e.reward for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.stack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.stack([e.done for e in experiences if e is not None]).astype(
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

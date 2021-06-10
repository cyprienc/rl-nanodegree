import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import ActorCriticNetwork

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
ACTOR_LR = 1e-3  # learning rate
CRITIC_LR = 1e-3  # learning rate
OU_THETA = 0.15  # Ornstein-Uhlenbeck process theta
OU_SIGMA = 0.2  # Ornstein-Uhlenbeck process sigma
UPDATE_PER_TS = 2  # how many times to update the network per timestep

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.actor_critic_local = ActorCriticNetwork(state_size, action_size, seed).to(
            device
        )
        self.actor_critic_target = ActorCriticNetwork(state_size, action_size, seed).to(
            device
        )
        self.actor_critic_target.load_state_dict(self.actor_critic_local.state_dict())

        self.critic_optimizer = optim.Adam(
            self.actor_critic_local.critic_parameters,
            lr=CRITIC_LR,
        )
        self.policy_optimizer = optim.Adam(
            self.actor_critic_local.actor_parameters,
            lr=ACTOR_LR,
        )
        self.critic_loss = nn.MSELoss()

        self.ou_process = np.zeros((1, action_size))

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        self.memory.add(states, actions, rewards, next_states, dones)

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > BATCH_SIZE:
            for i in range(UPDATE_PER_TS):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, states, train_mode):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            train_mode (bool): if True, adds noise to the actions
        """
        states = torch.from_numpy(states).float().to(device)
        self.actor_critic_local.eval()
        with torch.no_grad():
            actions = self.actor_critic_local.actor(states)
        self.actor_critic_local.train()

        if train_mode:
            if actions.size() != self.ou_process.size:
                self.ou_process = np.zeros(actions.size())

            self.ou_process += -OU_THETA * self.ou_process + np.random.normal(
                loc=0, scale=OU_SIGMA, size=actions.size()
            )
            return np.clip(
                actions.cpu().data.numpy() + self.ou_process, a_min=-1, a_max=1
            )
        else:
            return actions.cpu().data.numpy()

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ----- Critic loss ----- #

        with torch.no_grad():
            next_actions = self.actor_critic_target.actor(next_states)
            y = rewards + (1 - dones) * gamma * self.actor_critic_target.critic(
                next_states, next_actions
            )
        predictions = self.actor_critic_local.critic(states, actions)
        critic_loss = self.critic_loss(predictions, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.actor_critic_local.critic_parameters, 1)

        self.critic_optimizer.step()

        # ----- Policy loss ----- #

        policy_loss = -self.actor_critic_local.critic(
            states, self.actor_critic_local.actor(states)
        ).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic_local.actor_parameters, 1)
        self.policy_optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.actor_critic_local, self.actor_critic_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)

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
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

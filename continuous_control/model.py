import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticNetwork(nn.Module):
    """Actor-Critic Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(ActorCriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Feature map
        self.bn1 = nn.BatchNorm1d(num_features=state_size)
        self.dense1 = nn.Linear(in_features=state_size, out_features=5 * state_size)

        # Actor
        self.bn2 = nn.BatchNorm1d(num_features=5 * state_size)
        self.actor_dense = nn.Linear(
            in_features=5 * state_size, out_features=5 * state_size
        )
        self.action_dense = nn.Linear(
            in_features=5 * state_size, out_features=action_size
        )

        # Critic
        self.critic_dense = nn.Linear(
            in_features=(5 * state_size) + action_size, out_features=5 * state_size
        )
        self.value_dense = nn.Linear(in_features=5 * state_size, out_features=1)

        self.feature_parameters = (
            list(self.dense1.parameters())
            + list(self.bn1.parameters())
            + list(self.bn2.parameters())
        )

        self.actor_parameters = list(self.actor_dense.parameters()) + list(
            self.action_dense.parameters()
        )

        self.critic_parameters = (
            self.feature_parameters
            + list(self.critic_dense.parameters())
            + list(self.value_dense.parameters())
        )

    def feature(self, state):
        """Maps state to abstract features used by both the actor and the critic"""
        features = self.bn1(state)
        features = F.relu(self.dense1(features))
        return self.bn2(features)

    def actor(self, state):
        """Maps state -> action values"""
        features = self.feature(state)

        features = F.relu(self.actor_dense(features))

        return F.tanh(self.action_dense(features))

    def critic(self, state, action):
        """Maps (state,action) -> (state,action) values"""
        features = self.feature(state)
        features = torch.cat((features, action), dim=1)
        features = F.relu(self.critic_dense(features))

        return self.value_dense(features)

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorNetwork(nn.Module):
    def __init__(
        self,
        agent_obs_space_size: int,
        agent_action_space_size: int,
        total_obs_space_size: int,
        total_action_space_size: int,
    ):
        """
        Actor-Critic network

        Critic is centralized: it receives the observations and actions of all the agents

        Args:
            agent_obs_space_size: agent's observation space size
            agent_action_space_size: agent's action space size
            total_obs_space_size: total observation space size
            total_action_space_size: total action space size
        """
        super().__init__()
        self.agent_obs_space_size = agent_obs_space_size
        self.agent_action_space_size = agent_action_space_size
        self.total_obs_space_size = total_obs_space_size
        self.total_action_space_size = total_action_space_size

        # Actor
        self.actor_input_bn = nn.BatchNorm1d(num_features=agent_obs_space_size).to(
            device
        )
        self.actor_feature_layer = nn.Linear(
            in_features=agent_obs_space_size, out_features=2 * agent_obs_space_size
        ).to(device)
        self.actor_feature_bn = nn.BatchNorm1d(
            num_features=2 * agent_obs_space_size
        ).to(device)

        self.actor_layer_1 = nn.Linear(
            in_features=2 * agent_obs_space_size,
            out_features=2 * agent_obs_space_size,
        ).to(device)

        self.actor_layer_2 = nn.Linear(
            in_features=2 * agent_obs_space_size,
            out_features=agent_action_space_size,
        ).to(device)

        # Critic
        self.critic_input_bn = nn.BatchNorm1d(num_features=total_obs_space_size).to(
            device
        )
        self.critic_feature_layer = nn.Linear(
            in_features=total_obs_space_size, out_features=2 * total_obs_space_size
        ).to(device)
        self.critic_feature_bn = nn.BatchNorm1d(
            num_features=2 * total_obs_space_size
        ).to(device)

        self.critic_layer_1 = nn.Linear(
            in_features=2 * total_obs_space_size + total_action_space_size,
            out_features=2 * total_obs_space_size,
        ).to(device)
        self.critic_layer_2 = nn.Linear(
            in_features=2 * total_obs_space_size, out_features=1
        ).to(device)

        # Parameters
        self.actor_params = (
            list(self.actor_input_bn.parameters())
            + list(self.actor_feature_layer.parameters())
            + list(self.actor_feature_bn.parameters())
            + list(self.actor_layer_1.parameters())
            + list(self.actor_layer_2.parameters())
        )

        self.critic_params = (
            list(self.critic_input_bn.parameters())
            + list(self.critic_feature_layer.parameters())
            + list(self.critic_feature_bn.parameters())
            + list(self.critic_layer_1.parameters())
            + list(self.critic_layer_2.parameters())
        )

    def actor(self, state: torch.Tensor) -> torch.Tensor:
        features = self.actor_input_bn(state)
        features = F.relu(self.actor_feature_layer(features))
        features = self.actor_feature_bn(features)
        x = F.relu(self.actor_layer_1(features))
        return torch.tanh(self.actor_layer_2(x))

    def critic(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = self.critic_input_bn(states)
        x = F.relu(self.critic_feature_layer(x))
        x = self.critic_feature_bn(x)
        x = torch.cat((x, actions), dim=1)
        x = F.relu(self.critic_layer_1(x))
        return self.critic_layer_2(x)

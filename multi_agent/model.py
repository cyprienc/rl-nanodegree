from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiAgentActorCriticNetwork(nn.Module):
    # Note: this architecture makes policy ensemble tricky

    def __init__(self, num_agents: int, state_size: int, action_space_size: int):
        super().__init__()
        self.num_agents = num_agents

        # Input Batch Normalization for each of the agents observations
        self.input_bn = [
            nn.BatchNorm1d(num_features=state_size) for _ in range(num_agents)
        ]

        # Feature map linear layers
        self.feature_layer = [
            nn.Linear(in_features=state_size, out_features=3 * state_size)
            for _ in range(num_agents)
        ]
        self.feature_bn = [
            nn.BatchNorm1d(num_features=3 * state_size) for _ in range(num_agents)
        ]

        # Action linear layers
        self.action_layer_1 = [
            nn.Linear(in_features=3 * state_size, out_features=3 * state_size)
            for _ in range(num_agents)
        ]
        # outputs the logits
        self.action_layer_2 = [
            nn.Linear(in_features=3 * state_size, out_features=action_space_size)
            for _ in range(num_agents)
        ]

        # Critics linear layers
        self.critic_layer_1 = [
            nn.Linear(
                in_features=3 * state_size + num_agents, out_features=3 * state_size
            )
            for _ in range(num_agents)
        ]
        # outputs the logits
        self.critic_layer_2 = [
            nn.Linear(in_features=3 * state_size, out_features=1)
            for _ in range(num_agents)
        ]

        # noinspection PyTypeChecker
        self.feature_parameters = [
            param
            for layer in self.input_bn + self.feature_layer + self.feature_bn
            for param in layer.parameters()
        ]

        self.critic_parameters = self.feature_parameters + [
            param
            for layer in self.critic_layer_1 + self.critic_layer_2
            for param in layer.parameters()
        ]

        self.actor_parameters = [
            param
            for layer in self.action_layer_1 + self.action_layer_2
            for param in layer.parameters()
        ]

    def features(self, states: torch.Tensor) -> List[torch.Tensor]:
        # Each agent is independent
        states = states.unbind(dim=1)
        # batch normalization
        features = (bn(obs) for obs, bn in zip(states, self.input_bn))
        # dense layer + ReLu
        features = (
            F.relu(feature_layer(f))
            for f, feature_layer in zip(features, self.feature_layer)
        )
        # extra batch normalization
        features = [bn(f) for f, bn in zip(features, self.feature_bn)]
        return features

    def actors(self, states: torch.Tensor) -> torch.Tensor:
        features = self.features(states)
        # dense layer + ReLu
        x = (
            F.relu(action_layer(f))
            for f, action_layer in zip(features, self.action_layer_1)
        )
        # dense layer for Q values
        x = [action_layer(x_i) for x_i, action_layer in zip(x, self.action_layer_2)]
        return torch.stack(x, dim=1)

    def critic(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        features = self.features(states)
        # Adding actions to features
        features = (torch.cat((f, actions), dim=1) for f in features)
        # dense layer + ReLu
        x = (
            F.relu(critic_layer(f))
            for f, critic_layer in zip(features, self.critic_layer_1)
        )
        # dense layer for Q values
        x = [critic_layer(x_i) for x_i, critic_layer in zip(x, self.critic_layer_2)]
        return torch.stack(x, dim=1)

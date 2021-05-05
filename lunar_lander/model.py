import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dense1 = nn.Linear(in_features=state_size, out_features=5 * state_size)
        self.dense2 = nn.Linear(in_features=5 * state_size, out_features=5 * state_size)
        self.dense3 = nn.Linear(in_features=5 * state_size, out_features=action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        q = F.relu(self.dense1(state))
        q = F.relu(self.dense2(q))
        return self.dense3(q)

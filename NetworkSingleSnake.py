import torch
import torch.nn as nn
import torch.optim as optim

class Qnetwork(nn.Module):
    """
    Q-network for approximating Q-values in reinforcement learning.

    Attributes:
    - input_dim (int): Dimensionality of the input state.
    - fc1_dim (int): Dimensionality of the first fully connected layer.
    - fc2_dim (int): Dimensionality of the second fully connected layer.
    - n_action (int): Number of possible actions.
    - lr (float): Learning rate for the optimizer.
    - network (nn.Sequential): Neural network architecture.
    - optimizer (torch.optim.Optimizer): Optimizer for updating network parameters.
    - loss (nn.Module): Loss function for training the network.
    """

    def __init__(self, input_dim, fc1_dim, fc2_dim, n_action, lr):
        """
        Initialize the Q-network.

        Parameters:
        - input_dim (int): Dimensionality of the input state.
        - fc1_dim (int): Dimensionality of the first fully connected layer.
        - fc2_dim (int): Dimensionality of the second fully connected layer.
        - n_action (int): Number of possible actions.
        - lr (float): Learning rate for the optimizer.
        """
        super(Qnetwork, self).__init__()
        self.lr = lr
        self.network = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, n_action),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, state):
        """
        Forward pass through the Q-network.

        Parameters:
        - state (torch.Tensor): Input state for Q-value estimation.

        Returns:
        - actions (torch.Tensor): Estimated Q-values for each action.
        """
        actions = self.network(state)
        return actions

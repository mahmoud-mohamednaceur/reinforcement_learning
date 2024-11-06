"""
General Purpose
This code implements a neural network model that approximates Q-values in the context of reinforcement learning (RL). Q-values are used in Q-learning, a type of RL algorithm, to represent the expected future rewards for taking specific actions in a given state. The primary goal of the Q-network is to learn the best action to take in any given state based on experience.

Key Components Explained
Importing Libraries:

torch: The main PyTorch library for tensor computations.
torch.nn: Contains classes and functions to build neural networks.
torch.optim: Contains optimization algorithms to update model parameters.
Qnetwork Class:

Inherits from nn.Module: This base class is essential for creating neural network models in PyTorch.
Attributes:
input_dim: Specifies the dimensionality of the input state, which represents the environment's state in reinforcement learning.
fc1_dim: Defines the number of neurons in the first fully connected (dense) layer.
n_action: Represents the number of possible actions the agent can take.
lr: Learning rate for the optimizer, controlling how much the model parameters are updated during training.
network: A sequential model that defines the architecture of the neural network, which consists of:
An input layer (fully connected) that transforms the input state to the first hidden layer.
A ReLU activation function to introduce non-linearity.
An output layer that produces the estimated Q-values for each possible action.
optimizer: Adam optimizer is used to optimize the neural network parameters based on gradients calculated during training.
loss: Mean Squared Error (MSE) loss function is used to compute the difference between the predicted Q-values and the target Q-values during training.
Constructor (__init__ method):

Initializes the network architecture, optimizer, and loss function.
Takes parameters for input dimension, first fully connected layer's size, number of actions, and learning rate.
Forward Method:

Defines the forward pass through the network, where the input state is processed to produce Q-value estimates.
It takes the input state as a tensor and outputs the corresponding estimated Q-values for each action.


"""

import torch
import torch.nn as nn
import torch.optim as optim

class Qnetwork(nn.Module):
    """
    Q-network for approximating Q-values in reinforcement learning.

    Attributes:
    - input_dim (int): Dimensionality of the input state.
    - fc1_dim (int): Dimensionality of the first fully connected layer.
    - n_action (int): Number of possible actions.
    - lr (float): Learning rate for the optimizer.
    - network (nn.Sequential): Neural network architecture for Q-value estimation.
    - optimizer (torch.optim.Optimizer): Optimizer for updating network parameters.
    - loss (nn.Module): Loss function for training the network.
    """

    def __init__(self, input_dim, fc1_dim, n_action, lr):
        """
        Initialize the Q-network.

        Parameters:
        - input_dim (int): Dimensionality of the input state.
        - fc1_dim (int): Dimensionality of the first fully connected layer.
        - n_action (int): Number of possible actions.
        - lr (float): Learning rate for the optimizer.
        """
        # Initialize the parent class
        super(Qnetwork, self).__init__()

        # Store learning rate
        self.lr = lr

        # Define the network architecture using a sequential model
        self.network = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),  # First fully connected layer
            nn.ReLU(),                       # Activation function
            nn.Linear(fc1_dim, n_action)    # Output layer for Q-values
        )

        # Initialize the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Define the loss function for training
        self.loss = nn.MSELoss()  # Mean Squared Error Loss

    def forward(self, state):
        """
        Forward pass through the Q-network.

        Parameters:
        - state (torch.Tensor): Input state for Q-value estimation.

        Returns:
        - actions (torch.Tensor): Estimated Q-values for each action.
        """
        # Pass the input state through the network to get Q-values
        actions = self.network(state)
        return actions

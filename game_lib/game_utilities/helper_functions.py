import pygame
from enum import Enum
from collections import namedtuple
import matplotlib.pyplot as plt

from ml_lib.reinforcement_learning_agent_model import Agent

# Initialize Pygame
pygame.init()

# Set the font for the game display
font = pygame.font.SysFont("comicsans" , 50)

# Enum representing directions
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Named tuple for representing a point (x, y)
Point = namedtuple('Point', 'x, y')

# Color constants
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Size of each block in the game grid
BLOCK_SIZE = 20

# Speed of the game (frames per second)
SPEED = 50


def plot(scores, mean_scores):
    """
    Plot the training scores and mean scores over time.

    Parameters:
    - scores: List of scores obtained in each game.
    - mean_scores: List of mean scores calculated over multiple games.

    """
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Scores')
    plt.plot(mean_scores, label='Mean Scores')
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    plt.savefig('analysis_output_folder/analysis_images/mean_score_over_training_epochen.png')
    plt.show(block=False)
    plt.pause(.1)

def Create_agent(input_dim, dim1, dim2, n_actions, lr, butch_size, mem_size, gamma):
        """
        Create an instance of the Agent class with specified parameters.

        Parameters:
        - input_dim: Dimension of the input state.
        - dim1: Number of neurons in the first hidden layer.
        - dim2: Number of neurons in the second hidden layer.
        - n_actions: Number of possible actions.
        - lr: Learning rate for the agent.
        - butch_size: Batch size for training the agent's neural network.
        - mem_size: Size of the replay memory for experience replay.
        - gamma: Discount factor for future rewards.

        Returns:
        - Agent: An instance of the Agent class.
        """
        return Agent(input_dim, dim1, dim2, n_actions, lr, butch_size, mem_size, gamma)



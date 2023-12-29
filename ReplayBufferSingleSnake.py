# Replay Buffer
import numpy as np

class ReplayBuffer:
    """
    A replay buffer implementation for storing and sampling experiences in reinforcement learning.

    The replay buffer is used to store past experiences (state, action, reward, new_state, done)
    and provides a mechanism for sampling random batches of experiences for training.

    Attributes:
    - mem_size (int): Maximum size of the replay buffer.
    - state_mem (numpy.ndarray): Buffer for storing current states.
    - new_state_mem (numpy.ndarray): Buffer for storing new states.
    - action_mem (numpy.ndarray): Buffer for storing actions.
    - reward_mem (numpy.ndarray): Buffer for storing rewards.
    - done_mem (numpy.ndarray): Buffer for storing done flags.
    - mem_cntr (int): Counter for the number of stored experiences.
    - batch_size (int): Size of the batch to sample during training.
    """

    def __init__(self, mem_size, input_dim, n_actions, batch_size):
        """
        Initialize the replay buffer.

        Parameters:
        - mem_size (int): Maximum size of the replay buffer.
        - input_dim (int): Dimensionality of the state.
        - n_actions (int): Number of possible actions.
        - batch_size (int): Size of the batch to sample during training.
        """
        self.mem_size = mem_size
        self.state_mem = np.zeros((mem_size, input_dim))
        self.new_state_mem = np.zeros((mem_size, input_dim))
        self.action_mem = np.zeros((mem_size, n_actions))
        self.reward_mem = np.zeros((mem_size))
        self.done_mem = np.zeros((mem_size,))
        self.mem_cntr = 0
        self.batch_size = batch_size

    def store_action(self, state, new_state, action, reward, done):
        """
        Store an experience tuple in the replay buffer.

        Parameters:
        - state (numpy.ndarray): Current state.
        - new_state (numpy.ndarray): New state after taking an action.
        - action (numpy.ndarray): Action taken.
        - reward (float): Reward received.
        - done (float): Flag indicating whether the episode is done.
        """
        index = self.mem_cntr % self.mem_size
        self.state_mem[index] = state
        self.new_state_mem[index] = new_state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.done_mem[index] = done
        self.mem_cntr += 1

    def sample_mem(self):
        """
        Sample a batch of experiences from the replay buffer.

        Returns:
        - state (numpy.ndarray): Batch of current states.
        - new_state (numpy.ndarray): Batch of new states.
        - action (numpy.ndarray): Batch of actions.
        - reward (numpy.ndarray): Batch of rewards.
        - done (numpy.ndarray): Batch of done flags.
        - batch_indices (numpy.ndarray): Indices of the sampled batch in the replay buffer.
        """
        mem_empty = min(self.mem_size, self.mem_cntr)
        if mem_empty >= self.batch_size:
            batch_indices = np.random.choice(mem_empty, self.batch_size, replace=False)
        else:
            batch_indices = np.random.choice(mem_empty, self.batch_size, replace=True)

        state = self.state_mem[batch_indices]
        new_state = self.new_state_mem[batch_indices]
        action = self.action_mem[batch_indices]
        reward = self.reward_mem[batch_indices]
        done = self.done_mem[batch_indices]

        return state, new_state, action, reward, done, batch_indices

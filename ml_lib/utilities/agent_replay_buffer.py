"""
    General Purpose
    The purpose of the replay buffer is to store past experiences (represented as state, action, reward, next state, and done flag tuples) and sample random batches of these experiences during training. By sampling from a diverse set of past experiences, the model avoids overfitting to recent experiences, allowing it to learn more stable and generalized policies.

    Key Components Explained
    Attributes:

    mem_size: Maximum number of experiences the buffer can hold. Once full, new experiences overwrite the oldest ones.
    state_mem, new_state_mem, action_mem, reward_mem, done_mem: Arrays to store components of each experience tuple:
    state_mem: Stores the current states.
    new_state_mem: Stores the resulting states after an action is taken.
    action_mem: Stores the actions taken by the agent.
    reward_mem: Stores the rewards received for each action.
    done_mem: Stores flags indicating whether the episode has ended.
    mem_cntr: Tracks the total number of experiences stored and is used to manage the circular buffer.
    batch_size: Defines the number of samples (experiences) to retrieve during each training step.
    Constructor (__init__ method):

    Initializes the memory buffers for each component of an experience tuple (state, action, reward, new state, and done flag) as numpy arrays of size mem_size.
    Sets the maximum buffer size, state dimension, number of actions, and batch size.
    store_action Method:

    Stores a single experience tuple in the replay buffer.
    Uses mem_cntr % mem_size to store new experiences in a circular manner, where old experiences are overwritten once the buffer reaches its maximum size.
    Increments mem_cntr to keep track of the number of stored experiences.
    sample_mem Method:

    Samples a batch of experiences from the replay buffer.
    Randomly selects indices of experiences from the buffer up to the number of stored experiences (mem_cntr).
    Ensures random sampling without replacement unless there are fewer experiences than batch_size, in which case it samples with replacement.
    Returns the selected batch of states, actions, rewards, next states, and done flags, which are then used for training the Q-network.
    Usage Context
    In reinforcement learning, this replay buffer supports experience replay by storing past experiences and allowing random sampling during training. This approach allows the agent to learn from a broader range of experiences, reducing bias and improving learning stability. Specifically:

    The agent’s Q-network can be trained by sampling batches from the replay buffer instead of using only the most recent experience.
    This replay mechanism breaks the temporal correlation between experiences, which helps stabilize the learning process, particularly in complex environments where episodes may be long and actions can have delayed effects.
"""
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

import torch
import random 
import os 
from ml_lib.utilities.agent_deep_neural_network import Qnetwork
from ml_lib.utilities.agent_replay_buffer import ReplayBuffer

class Agent:
    """
    A reinforcement learning agent using Q-learning with experience replay.

    Attributes:
    - input_dim (int): Dimensionality of the input state.
    - fc1_dim (int): Dimensionality of the first fully connected layer.
    - fc2_dim (int): Dimensionality of the second fully connected layer.
    - n_actions (int): Number of possible actions.
    - lr (float): Learning rate for the neural network.
    - butch_size (int): Size of the batch for training.
    - mem_size (int): Size of the replay memory.
    - gamma (float): Discount factor for future rewards.
    - epsilon (float): Exploration-exploitation trade-off parameter.
    - n_games (int): Number of games played.
    - mem (ReplayBuffer): Replay buffer for storing experiences.
    - network (Qnetwork): Neural network for Q-value estimation.
    """

    def __init__(self, input_dim, fc1_dim, fc2_dim, n_actions, lr, butch_size, mem_size, gamma):
        """
        Initialize the agent.

        Parameters:
        - input_dim (int): Dimensionality of the input state.
        - fc1_dim (int): Dimensionality of the first fully connected layer.
        - fc2_dim (int): Dimensionality of the second fully connected layer.
        - n_actions (int): Number of possible actions.
        - lr (float): Learning rate for the neural network.
        - butch_size (int): Size of the batch for training.
        - mem_size (int): Size of the replay memory.
        - gamma (float): Discount factor for future rewards.
        """
        self.input_dim = input_dim
        self.fc1_dim = fc1_dim
        #self.fc2_dim = fc2_dim
        self.n_actions = n_actions
        self.lr = lr
        self.butch_size = butch_size
        self.mem_size = mem_size
        self.epsilon = 0
        self.gamma = gamma
        self.n_games = 0

        # Initialize replay buffer and neural network
        self.mem = ReplayBuffer(mem_size, input_dim, n_actions, butch_size)
        self.network = Qnetwork(input_dim, fc1_dim, n_actions, lr)

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy.

        Parameters:
        - state (list or numpy.ndarray): Current state.

        Returns:
        - final_move (list): Action represented as a one-hot vector.
        """
        self.epsilon = max(0, 80 - self.n_games)

        final_move = [0] * self.n_actions

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, self.n_actions - 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.network(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def short_mem(self, state, next_state, action, reward, done):
        """
        Store short-term memories in the replay buffer and learn from them.

        Parameters:
        - state (list or numpy.ndarray): Current state.
        - next_state (list or numpy.ndarray): Next state.
        - action (list or numpy.ndarray): Action taken.
        - reward (float): Reward received.
        - done (bool): Flag indicating whether the episode is done.
        """
        self.learn(state, next_state, action, reward, done)

    def long_mem(self):
        """
        Learn from experiences stored in the replay buffer.

        This method is typically called after accumulating enough experiences in the replay buffer.
        """
        if self.butch_size < self.mem.mem_cntr:
            return

        states, next_states, actions, rewards, dones, butch = self.mem.sample_mem()
        self.learn(states, next_states, actions, rewards, dones)

    def save(self, file_name='reinforcement_agent_trained_model.pth'):
        """
        Save the trained neural network's parameters to a file.

        Parameters:
        - file_name (str): Name of the file to save the parameters.
        """

        model_folder_path = '../analysis_output_folder/trained_model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.network.state_dict(), file_name)

    def learn(self, state, next_state, action, reward, done):
        """
        Update the neural network's parameters based on the Q-learning update rule.

        Parameters:
        - state (list or numpy.ndarray): Current state.
        - next_state (list or numpy.ndarray): Next state.
        - action (list or numpy.ndarray): Action taken.
        - reward (float): Reward received.
        - done (bool): Flag indicating whether the episode is done.
        """
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.network(state)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.network(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.network.optimizer.zero_grad()
        loss = self.network.loss(target, pred)
        loss.backward()
        self.network.optimizer.step()

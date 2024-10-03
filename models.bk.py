import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


# Define the Q-network model using TensorFlow/Keras
class DQN(tf.keras.Model):
    def __init__(self, state_size, action_size, hidden_size=64):
        """
        Initializes the Deep Q-Network model.
        
        Args:
        - state_size: Number of input features (size of the state space)
        - action_size: Number of actions (size of the action space)
        - hidden_size: Size of the hidden layers
        """
        super(DQN, self).__init__()
        self.fc1 = layers.Dense(hidden_size, activation='relu')  # First fully connected layer
        self.fc2 = layers.Dense(hidden_size, activation='relu')  # Second fully connected layer
        self.fc3 = layers.Dense(action_size)  # Output layer to predict Q-values

    def call(self, state):
        """
        Forward pass of the network.
        
        Args:
        - state: The input state (tensor)
        
        Returns:
        - Q-values for all possible actions
        """
        x = self.fc1(state)
        x = self.fc2(x)
        return self.fc3(x)  # Output Q-values (no activation function)

# Q-Learning Agent using the TensorFlow model
class QLearningAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99):
        """
        Initializes the Q-learning agent with a TensorFlow Q-network.
        
        Args:
        - state_size: Size of the state space
        - action_size: Size of the action space
        - lr: Learning rate for the optimizer
        - gamma: Discount factor for future rewards
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor
        
        # Initialize the Q-network (policy network)
        self.q_network = DQN(state_size, action_size)
        self.q_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                                loss='mse')  # Compile the model with Adam optimizer and MSE loss

    def get_action(self, state, epsilon):
        """
        Selects an action using an epsilon-greedy policy.
        
        Args:
        - state: Current state of the environment (numpy array)
        - epsilon: Exploration rate (probability of choosing a random action)
        
        Returns:
        - action: The selected action (int)
        """
        if np.random.rand() < epsilon:
            # Take a random action (exploration)
            return np.random.randint(0, self.action_size)
        else:
            # Take the action with the highest Q-value (exploitation)
            state = np.expand_dims(state, axis=0)  # Add batch dimension
            q_values = self.q_network.predict(state, verbose=0)
            return np.argmax(q_values[0])

    def choose_action(self, state):
        """
        Selects an action using a greedy policy (no exploration).
        
        Args:
        - state: Current state of the environment (numpy array)
        
        Returns:
        - action: The selected action (int)
        """
        state = np.expand_dims(state, axis=0)
        q_values = self.q_network.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        """
        Performs one step of Q-learning by updating the Q-network.
        
        Args:
        - state: The current state (numpy array)
        - action: The action taken (int)
        - reward: The reward received (float)
        - next_state: The next state after taking the action (numpy array)
        - done: Whether the episode has ended (boolean)
        """
        # Prepare the state and next state as batched inputs
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        
        # Predict the Q-values for the current state
        q_values = self.q_network.predict(state, verbose=0)
        
        # Predict the Q-values for the next state
        next_q_values = self.q_network.predict(next_state, verbose=0)
        
        # Compute the target Q-value
        target_q_value = reward + (self.gamma * np.max(next_q_values[0]) * (1 - done))
        
        # Update the Q-value for the selected action
        q_values[0][action] = target_q_value
        
        # Perform the training step (minimize the loss between predicted and target Q-values)
        self.q_network.fit(state, q_values, verbose=0)

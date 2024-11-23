# agent_template.py

import gymnasium as gym
from state_discretizer import StateDiscretizer
import numpy as np
import pickle

class LunarLanderAgent:
    def __init__(self):
        """
        Initialize your agent here.

        This method is called when you create a new instance of your agent.
        Use this method to Initializes the environment, the agent’s model (e.g., Q-table or neural network),
        and the optional state discretizer for Q-learning. Add any necessary initialization for model parameters here.
        """
        # Initialize environment
        self.env = gym.make('LunarLander-v3')

        # Initialize state discretizer if you are going to use Q-Learning
        self.state_discretizer = StateDiscretizer(self.env)

        # Initialize Q-table
        self.q_table = [np.zeros(self.state_discretizer.iht_size) for _ in range(self.env.action_space.n)]

        # Set learning parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.995  # Exploration decay rate
        # self.epsilon_min = 0.01  # Minimum exploration rate

        # # Initialize training mode
        # self.training = True

    def select_action(self, state):
        """
        Given a state, select an action to take. The function should operate in training and testing modes,
        where in testing you will need to shut off epsilon-greedy selection.

        Args:
            state (array): The current state of the environment.

        Returns:
            int: The action to take.
        """
        if np.random.uniform(0, 1) < self.epsilon:  # Exploration
            return self.env.action_space.sample()  # Random action
        else:  # Exploitation
            # Discretize the state
            state_features = self.state_discretizer.discretize(state)
            # Compute Q-values for each action
            q_values = [self.q_table[action][state_features].sum() for action in range(self.env.action_space.n)]
            return int(np.argmax(q_values))  # Select action with highest Q-value

    def train(self, num_episodes):
        """
         Contains the main training loop where the agent learns over multiple episodes.

        Args:
            num_episodes (int): Number of episodes to train for.
        """
        best_avg_reward = -float('inf')
        rewards = []

        for episode in range(num_episodes):
            state = self.env.reset()[0]
            done = False
            total_reward = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            rewards.append(total_reward)

            self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)  # Decay epsilon

            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                print(f"Episode {episode}, Avg Reward (last 100): {avg_reward}")

    def update(self, state, action, reward, next_state, done):
        """
        Update your agent's knowledge based on the transition.

        Args:
            state (array): The previous state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (array): The new state after the action.
            done (bool): Whether the episode has ended.
        """
        state_features = self.state_discretizer.discretize(state)
        next_state_features = self.state_discretizer.discretize(next_state)
    # Q-learning update rule
        max_next_q = max([self.q_table[a][next_state_features].sum() for a in range(self.env.action_space.n)])
        target = reward + (0 if done else self.gamma * max_next_q)
        self.q_table[action][state_features] += self.alpha * (target - self.q_table[action][state_features].sum())

    def test(self, num_episodes=100):
        """
        Test your agent locally before submission to get a hint of the expected score.

        Args:
            num_episodes (int): Number of episodes to test for.
        """
        total_rewards = []
        self.epsilon = 0  # Disable exploration during testing

        for episode in range(num_episodes):
            state = self.env.reset()[0]
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                state, reward, done, _, _ = self.env.step(action)
                total_reward += reward

            total_rewards.append(total_reward)

        avg_reward = np.mean(total_rewards)
        print(f"Test completed. Average Reward: {avg_reward}")
        return avg_reward

    def save_agent(self, file_name):
        """
        Save your agent's model to a file.

        Args:
            file_name (str): The file name to save the model.
        """
        with open(file_name, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'iht_dict': self.state_discretizer.iht.dictionary
            }, f)
        print(f"Model saved to {file_name}.")

    def load_agent(self, file_name):
        """
        Load your agent's model from a file.

        Args:
            file_name (str): The file name to load the model from.
        """
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.state_discretizer.iht.dictionary = data['iht_dict']
        print(f"Model loaded from {file_name}.")

if __name__ == '__main__':

    agent = LunarLanderAgent()
    agent_model_file = 'model.pkl'  # Set the model file name

    # Example usage:
    # Uncomment the following lines to train your agent and save the model

    num_training_episodes = 1000  # Define the number of training episodes
    print("Training the agent...")
    agent.train(num_training_episodes)
    print("Training completed.")

    # Save the trained model
    agent.save_agent(agent_model_file)
    print("Model saved.")

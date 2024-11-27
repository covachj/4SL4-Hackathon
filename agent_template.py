# agent_template.py

import gymnasium as gym
from state_discretizer import StateDiscretizer
import numpy as np
import pickle
import random

agent_model_file = 'model2.pkl'  # Set the model file name

class LunarLanderAgent:
    def __init__(self):
        """
        Initialize your agent here.

        This method is called when you create a new instance of your agent.
        Use this method to Initializes the environment, the agent’s model (e.g., Q-table or neural network),
        and the optional state discretizer for Q-learning. Add any necessary initialization for model parameters here.
        """
        # TODO: Initialize your agent's parameters and variables

        # Initialize environment
        self.env = gym.make('LunarLander-v3')

        # Initialize state discretizer if you are going to use Q-Learning
        self.state_discretizer = StateDiscretizer(self.env)

        # initialize Q-table or neural network weights
        self.num_actions = self.env.action_space.n
        self.q_table = [np.zeros(self.state_discretizer.iht_size) for _ in range(self.num_actions)]

        # Set learning parameters
        alpha = 0.5
        self.alpha = alpha / self.state_discretizer.num_tilings  # Learning rate per tiling
        self.epsilon =   1.0      # Initial exploration rate
        self.epsilon_decay = 0.995 # Exploration decay rate
        self.min_epsilon = 0.01   # Minimum exploration rate
        self.gamma = 0.99
        # Initialize any other parameters and variables
        self.total_steps = 0  # To keep track of total steps taken


    def select_action(self, state,test_mode= True):
        """
        Given a state, select an action to take. The function should operate in training and testing modes,
        where in testing you will need to shut off epsilon-greedy selection.

        Args:
            state (array): The current state of the environment.

        Returns:
            int: The action to take.
        """
        # TODO: Implement your action selection policy here
        # For example, you might use an epsilon-greedy policy if you're using Q-learning
        # Ensure the action returned is an integer in the range [0, 3]
         
        # Discretize the state if you are going to use Q-Learning
        state_features = self.state_discretizer.discretize(state)
        Q_values = [self.q_table[action][state_features].sum() for action in range(self.num_actions)]

        # Epsilon-greedy action selection
        if not test_mode and np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            # Select action with highest Q-value
            max_Q = np.max(Q_values)
            max_actions = [action for action, Q in enumerate(Q_values) if Q == max_Q]
            action = np.random.choice(max_actions)

        return action

    def train(self, num_episodes):
        """
         Contains the main training loop where the agent learns over multiple episodes.

        Args:
            num_episodes (int): Number of episodes to train for.
        """
        # TODO: Implement your training loop here
        # Make sure to:
        # 1) Evaluate the training in each episode by monitoring the average of the previous ~100
        #    episodes cumulative rewards (return).
        # 2) Autosave the best model achived in each epoch based on the evaluation.
        best_avg_reward = -float('inf')
        rewards_history = []
        terminated = False
        for episode in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            done = False
            truncated = False  # Initialize truncated flag
            terminated = False  # Reset terminated at the start of each episode
            total_reward = 0

            while not terminated:
                # Set test_mode=False during training to enable exploration
                action = self.select_action(state, test_mode=False)
                next_state, reward, done, truncated, info = self.env.step(action)
                terminated = done or truncated
                self.update(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward
                self.total_steps += 1

            # Decay epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

            rewards_history.append(total_reward)

            # Evaluate performance over last 100 episodes
            if episode >= 100:
                avg_reward = np.mean(rewards_history[-100:])
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    self.save_agent(agent_model_file)
                    print(f'Episode {episode}: New best average reward {best_avg_reward:.2f}, Reward {total_reward:.2f}, model saved.')
                else:
                    print(f'Episode {episode}: Average reward {avg_reward:.2f} Epsilon {self.epsilon:.4f}, Reward {total_reward:.2f}')
            else:
                print(f'Episode {episode}: Reward {total_reward:.2f} Epsilon {self.epsilon:.4f}')
   

           

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
        # TODO: Implement your agent's update logic here
        # This method is where you would update your Q-table or neural network

        # Discretize the states if you are going to use Q-Learning
        state_features = self.state_discretizer.discretize(state)
        next_state_features = self.state_discretizer.discretize(next_state)

        # Compute current Q-value
        current_Q = self.q_table[action][state_features].sum()

        # Compute TD target
        if done:
            target = reward
        else:
            Q_values_next = [self.q_table[next_action][next_state_features].sum() for next_action in range(self.num_actions)]
            max_Q_next = max(Q_values_next)
            target = reward + self.gamma * max_Q_next

        # Compute TD error
        delta = target - current_Q

        # Update Q-table weights for the taken action and active features
        self.q_table[action][state_features] += self.alpha * delta

    
    def test(self, num_episodes = 100):
        """
        Test your agent locally before submission to get a hint of the expected score.

        Args:
            num_episodes (int): Number of episodes to test for.
        """
        # TODO: Implement your testing loop here
        # Make sure to:
        # Store the cumulative rewards (return) in all episodes and then take the average 
        total_rewards = []

        for episode in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                # y Bdefault, select_action operates in test_mode=True
                action = self.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                state = next_state
                total_reward += reward

            total_rewards.append(total_reward)
            print(f'Test Episode {episode}: Reward {total_reward:.2f}')

        avg_reward = np.mean(total_rewards)
        print(f'Average Reward over {num_episodes} episodes: {avg_reward:.2f}')


    def save_agent(self, file_name):
        """
        Save your agent's model to a file.

        Args:
            file_name (str): The file name to save the model.
        """
        # TODO: Implement code to save your model (e.g., Q-table, neural network weights)
        # Example: for Q-learining:
        with open(file_name, 'wb') as f:
          pickle.dump({
              'q_table': self.q_table,
              'iht_dict': self.state_discretizer.iht.dictionary
          }, f)

    def load_agent(self, file_name):
        """
        Load your agent's model from a file.

        Args:
            file_name (str): The file name to load the model from.
        """
        # TODO: Implement code to load your model
        # Example: for Q-learining:
        with open(file_name, 'rb') as f:
           data = pickle.load(f)
           self.q_table = data['q_table']
           self.state_discretizer.iht.dictionary = data['iht_dict']
        print(f"Model loaded from {file_name}.")


if __name__ == '__main__':

    agent = LunarLanderAgent()

    # Example usage:
    # Uncomment the following lines to train your agent and save the model

    num_training_episodes = 1000  # Define the number of training episodes
    print("Training the agent...")
    agent.train(num_training_episodes)
    print("Training completed.")

    # Save the trained model
    agent.save_agent(agent_model_file)
    print("Model saved.")

    # Test the agent
    print("Testing the agent...")
    agent.load_agent(agent_model_file)
    agent.test(num_episodes=100)
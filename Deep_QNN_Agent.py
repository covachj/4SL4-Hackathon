# agent_template.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import random
from torch import nn
from torch import optim
import torch.nn.functional as F
import gymnasium as gym
from state_discretizer import StateDiscretizer
from collections import namedtuple , deque
from itertools import count



device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


## ************************** Experience replay, Learned from PyTorch Documentation **************************

#Mapping of pairs to their resulting reward and state
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


#Memory Buffer
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


## **********************************************************************************************************

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, n_actions)

    #Forward Prop
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)



class LunarLanderAgent():
    def __init__(self):
        """
        Initialize your agent here.

        This method is called when you create a new instance of your agent.
        Use this method to Initializes the environment, the agent’s model (e.g., Q-table or neural network),
        and the optional state discretizer for Q-learning. Add any necessary initialization for model parameters here.
        """

        # Initialize environment
        self.env = gym.make('LunarLander-v3')
        self.n_actions = self.env.action_space.n
        self.state, self.info = self.env.reset()
        self.n_observations = len(self.state)

        # Initialize the network
        self.policy = DQN(self.n_observations, self.n_actions).to(device)
        self.target = DQN(self.n_observations, self.n_actions).to(device)
        self.target.load_state_dict(self.policy.state_dict())


        # Set parameters & Utils
        self.BATCH_SIZE = 64
        self.DISCOUNT_FACTOR = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4
        self.optimizer = optim.AdamW(self.policy.parameters(), lr = self.LR, amsgrad = True)
        self.memory = ReplayMemory(10000)
        self.steps_complete = 0
        self.episode_durations = []
        self.avg_rewards = []
    
    def select_action(self, state, TRAIN = False):
        """
        Given a state, select an action to take. The function should operate in training and testing modes,
        where in testing you will need to shut off epsilon-greedy selection.

        Args:
            state (array): The current state of the environment.

        Returns:
            int: The action to take.
        """

        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
        math.exp(-1. * self.steps_complete / self.EPS_DECAY)
        self.steps_complete += 1
        if sample < eps_threshold and TRAIN:
            return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
              # t.max(1) will return the largest column value of each row.
              # second column on max result is index of where max element was
              # found, so we pick action with the larger expected reward.
              return self.policy(state).max(1).indices.view(1, 1).numpy()[0,0]
            
        
    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated


    def optimize(self):
      if len(self.memory) < self.BATCH_SIZE:
          return
      transitions = self.memory.sample(self.BATCH_SIZE)
      batch = Transition(*zip(*transitions))
      non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
      non_final_next_states = torch.cat([s for s in batch.next_state
                                                  if s is not None])
      state_batch = torch.cat(batch.state)
      action_batch = torch.cat(batch.action)
      reward_batch = torch.cat(batch.reward)

      # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
      # columns of actions taken. These are the actions which would've been taken
      # for each batch state according to policy_net
      state_action_values = self.policy(state_batch).gather(1, action_batch)

      # Compute V(s_{t+1}) for all next states.
      # Expected values of actions for non_final_next_states are computed based
      # on the "older" target_net; selecting their best reward with max(1).values
      # This is merged based on the mask, such that we'll have either the expected
      # state value or 0 in case the state was final.
      next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
      with torch.no_grad():
          next_state_values[non_final_mask] = self.target(non_final_next_states).max(1).values
      # Compute the expected Q values
      expected_state_action_values = (next_state_values * self.DISCOUNT_FACTOR) + reward_batch

      # Compute Huber loss
      criterion = nn.SmoothL1Loss()
      loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

      # Optimize the model
      self.optimizer.zero_grad()
      loss.backward()
      # In-place gradient clipping
      torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
      self.optimizer.step()


    def train(self, num_episodes):
        """
         Contains the main training loop where the agent learns over multiple episodes.

        Args:
            num_episodes (int): Number of episodes to train for.
        """
        rewards = deque([], maxlen=100)
        top_reward = 0
        for i_episode in range(num_episodes):
            # Initialize the environment and get its state
            self.state, self.info = self.env.reset()
            reward_sum = 0
            for t in count():
                action = torch.tensor(np.reshape(np.asarray(self.select_action(self.state, TRAIN = True)), (1,1)))
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward_sum += reward
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(torch.tensor(np.reshape(self.state, (1,-1))), action, next_state, reward)

                # Move to the next state
                self.state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target.state_dict()
                policy_net_state_dict = self.policy.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    # self.plot_durations()
                    break
                
                self.state = np.reshape(self.state.numpy(), (1,-1))[0]
            print(f"Reward = {reward_sum}, EP = {i_episode}")
            rewards.append(reward_sum)
            avg = sum(rewards)/100
            self.avg_rewards.append(avg)
            if avg > top_reward:
                top_reward = avg
                print(f"Model Written, Reward: {top_reward}")
                self.save_agent("DQN.pth")

        print('Complete')



    # def update(self, state, action, reward, next_state, done):
    #     """
    #     Update your agent's knowledge based on the transition.

    #     Args:
    #         state (array): The previous state.
    #         action (int): The action taken.
    #         reward (float): The reward received.
    #         next_state (array): The new state after the action.
    #         done (bool): Whether the episode has ended.
    #     """
    #     # TODO: Implement your agent's update logic here
    #     # This method is where you would update your Q-table or neural network

    #     # Discretize the states if you are going to use Q-Learning
    #     # state_features = self.state_discretizer.discretize(state)
    #     # next_state_features = self.state_discretizer.discretize(next_state)

    #     pass
    
    def test(self, num_episodes = 100):
        """
        Test your agent locally before submission to get a hint of the expected score.

        Args:
            num_episodes (int): Number of episodes to test for.
        """
        total_rewards = []

        for episode in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                # y Bdefault, select_action operates in test_mode=True
                action = torch.tensor(np.reshape(np.asarray(self.select_action(state, TRAIN = True)), (1,1)))
                next_state, reward, terminated, truncated, info = self.env.step(action.item())
                done = terminated or truncated
                if terminated:
                    next_state = None

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
        torch.save(self.policy.state_dict(), file_name)

    def load_agent(self, file_name):
        """
        Load your agent's model from a file.

        Args:
            file_name (str): The file name to load the model from.
        """
        self.policy.load_state_dict(torch.load(file_name, weights_only=True))
        print(f"Model loaded from {file_name}.")

if __name__ == '__main__':

    agent = LunarLanderAgent()
    agent_model_file = 'DQN.pth'  # Set the model file name
    
    # Example usage:
    # Uncomment the following lines to train your agent and save the model

    num_training_episodes = 1000 # Define the number of training episodes
    print("Training the agent...")
    agent.train(num_training_episodes)
    print("Training completed.")
    plt.plot(range(num_training_episodes),agent.avg_rewards)
    plt.title("Avg cumulative Reward")
    plt.show()
    agent.load_agent(agent_model_file)
    agent.test()

    # Save the trained model
    # agent.save_model(agent_model_file)
    # print("Model saved.")

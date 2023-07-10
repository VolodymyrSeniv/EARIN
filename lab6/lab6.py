import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt

class QLearning:
    def __init__(self, env, alpha, gamma, epsilon, number_episodes, num_bins, lower_bounds, upper_bounds):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_number = env.action_space.n
        self.number_episodes = number_episodes
        self.num_bins = num_bins
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.sum_rewards_episode = [] # List to store the sum of rewards for each episode
        self.Q_matrix = np.random.uniform(low=0, high=1, size=(*num_bins, self.action_number)) # Q-matrix initialization

    def return_index_state(self, state):
        indices = []
        for i in range(len(state)):
            # Mapping the continuous state to a discrete index based on binning
            indices.append(np.maximum(
                np.digitize(state[i], np.linspace(self.lower_bounds[i], self.upper_bounds[i], self.num_bins[i])) - 1,
                0))
        return tuple(indices) # Returning the tuple of indices

    def select_action(self, state, index):
        if index < 500:
            return np.random.choice(self.action_number) # Random action selection for initial episodes

        random_number = np.random.random()
        if index > 7000:
            self.epsilon = 0.999 * self.epsilon # Decay epsilon over time
        if random_number < self.epsilon:
            return np.random.choice(self.action_number) # Exploration: Random action selection

        else:
            return np.random.choice(np.where(
                self.Q_matrix[self.return_index_state(state)] == np.max(self.Q_matrix[self.return_index_state(state)]))[
                                        0]) # Exploitation: Action selection based on Q-values

    def simulate_episodes(self):
        for index_episode in range(self.number_episodes):
            rewards_episode = [] # List to store rewards for the current episode
            state_s, _ = self.env.reset()
            state_s = list(state_s)
            print("Simulating episode {}".format(index_episode))

            terminal_state = False
            while not terminal_state:
                state_s_index = self.return_index_state(state_s)
                action_a = self.select_action(state_s, index_episode)
                state_s_prime, reward, terminal_state, _, _ = self.env.step(action_a)
                rewards_episode.append(reward)
                state_s_prime = list(state_s_prime)
                state_s_prime_index = self.return_index_state(state_s_prime)
                Q_max_prime = np.max(self.Q_matrix[state_s_prime_index])

                if not terminal_state:
                    error = reward + self.gamma * Q_max_prime - self.Q_matrix[state_s_index + (action_a,)]
                    self.Q_matrix[state_s_index + (action_a,)] += self.alpha * error # Q-value update for non-terminal state
                else:
                    error = reward - self.Q_matrix[state_s_index + (action_a,)]
                    self.Q_matrix[state_s_index + (action_a,)] += self.alpha * error # Q-value update for terminal state
                state_s = state_s_prime
            print("Sum of rewards {}".format(np.sum(rewards_episode)))
            self.sum_rewards_episode.append(np.sum(rewards_episode)) # Store the sum of rewards for the current episode

    def simulate_learned_strategy(self):
        env1 = gym.make('CartPole-v1', render_mode='human')
        current_state, _ = env1.reset()
        env1.render()
        time_steps = 1000
        obtained_rewards = [] # List to store rewards obtained from the learned strategy

        for time_index in range(time_steps):
            print(time_index)
            action_in_state_s = np.random.choice(np.where(
                self.Q_matrix[self.return_index_state(current_state)] == np.max(
                    self.Q_matrix[self.return_index_state(current_state)]))[0]) # Action selection based on learned Q-values
            current_state, reward, terminated, truncated, info = env1.step(action_in_state_s)
            obtained_rewards.append(reward)
            time.sleep(0.05)
            if terminated:
                time.sleep(1)
                break
        return obtained_rewards, env1


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state, _ = env.reset()

    upper_bounds = env.observation_space.high
    lower_bounds = env.observation_space.low

    # Adjusting the upper and lower bounds of specific state variables
    cart_velocity_min = -3
    cart_velocity_max = 3
    pole_angle_velocity_min = -10
    pole_angle_velocity_max = 10
    upper_bounds[1] = cart_velocity_max # Upper bound for cart velocity
    upper_bounds[3] = pole_angle_velocity_max # Upper bound for pole angle velocity
    lower_bounds[1] = cart_velocity_min # Lower bound for cart velocity
    lower_bounds[3] = pole_angle_velocity_min # Lower bound for pole angle velocity

    num_bins_position = 30
    num_bins_velocity = 30
    num_bins_angle = 30
    num_bins_angle_velocity = 30
    num_bins = [num_bins_position, num_bins_velocity, num_bins_angle, num_bins_angle_velocity] # Number of bins for each state variable

    alpha = 0.01 # Learning rate
    gamma = 0.95 # Discount factor
    epsilon = 0.1 # Exploration rate
    number_episodes = 1000

    Q1 = QLearning(env, alpha, gamma, epsilon, number_episodes, num_bins, lower_bounds, upper_bounds)
    Q1.simulate_episodes()

    obtained_rewards_optimal, env1 = Q1.simulate_learned_strategy()

    plt.figure(figsize=(12, 5))
    plt.plot(Q1.sum_rewards_episode, color='blue', linewidth=1) # Plot the sum of rewards for each episode
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.yscale('log') # Set logarithmic scale on the y-axis
    plt.show()

    env1.close()

    np.sum(obtained_rewards_optimal) # Calculate the sum of rewards obtained from the learned strategy

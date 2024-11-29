
import os
import numpy as np
import matplotlib.pyplot as plt
from env import Environment  # Import the Environment class

class Agent:
    def __init__(self, epsilon, alpha, gamma):
        self.epsilon = epsilon  # Initial exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_values = {}  # Store Q-values for state-action pairs

    def choose_action(self, state, env):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(env.actions)  # Explore
        else:
            action = self.get_action_with_max_q(state, env)  # Exploit
        return action

    def update_q_value(self, state, action, reward, next_state, env):
        old_q_value = self.q_values.get((state, action), 0)
        next_max = max([self.q_values.get((next_state, a), 0) for a in env.actions])
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * next_max - old_q_value)
        self.q_values[(state, action)] = new_q_value
        return abs(new_q_value - old_q_value)  # Return the change for convergence tracking (optional)

    def get_action_with_max_q(self, state, env):
        q_values_state = [self.q_values.get((state, a), 0) for a in env.actions]
        max_q = max(q_values_state)
        best_actions = [i for i, q in enumerate(q_values_state) if q == max_q]
        action_index = np.random.choice(best_actions)
        return env.actions[action_index]

    def train(self, env, episodes):
        rewards_per_episode = []
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state, env)
                next_state, reward, done = env.step(action)
                self.update_q_value(state, action, reward, next_state, env)
                state = next_state
                total_reward += reward

            rewards_per_episode.append(total_reward)

            # Gradual epsilon decay (optional): slow down the decay rate for more exploration
            # Adjust this line based on your desired decay schedule
            self.epsilon = max(0.01, self.epsilon * 0.995)  # Smoother decay

        return rewards_per_episode

    def get_optimal_sequence_and_reward(self, env):
        state = env.reset()
        sequence = [state]
        done = False
        steps = 0
        while not done:
            action = self.get_action_with_max_q(state, env)
            next_state, reward, done = env.step(action)
            sequence.append(next_state)
            state = next_state
            steps += 1
        return sequence, steps

def smooth_rewards(rewards_per_episode, window_size=5000):
    smoothed_rewards = []
    for i in range(len(rewards_per_episode)):
        if i < window_size:
            smoothed_rewards.append(np.mean(rewards_per_episode[:i + 1]))
        else:
            smoothed_rewards.append(np.mean(rewards_per_episode[i - window_size:i + 1]))
    return smoothed_rewards


# Main
if __name__ == "__main__":
    episodes = 10000
    alpha_values = [0.1, 0.3, 0.5, 0.9]
    gamma_values = [0.9, 0.95, 0.99]
    epsilon_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    output_folder = "new_result"
    os.makedirs(output_folder, exist_ok=True)

    env = Environment()

    for gamma in gamma_values:
        for alpha in alpha_values:
            for epsilon in epsilon_values:
                agent = Agent(epsilon=epsilon, alpha=alpha, gamma=gamma)
                rewards_per_episode = agent.train(env, episodes=episodes)
                optimal_sequence, steps = agent.get_optimal_sequence_and_reward(env)
                smoothed_rewards = smooth_rewards(rewards_per_episode)
                
                # Print optimal sequence and number of steps
                print(f"Optimal Sequence (gamma={gamma}, alpha={alpha}, epsilon={epsilon}): {optimal_sequence}")
                print(f"Number of Steps: {steps}") 
                # Print Reward Progression Over Episodes
                print(f"Reward in the last episode: {smoothed_rewards[-1]}") 

                # Ensure the rewards list has the correct length (10,000 episodes)
                if len(smoothed_rewards) < episodes:
                    smoothed_rewards.extend([smoothed_rewards[-1]] * (episodes - len(smoothed_rewards)))

                # Plotting Reward Progression Over Episodes
                plt.figure(figsize=(10, 6))
                plt.plot(np.arange(episodes), smoothed_rewards, label="Total Reward", color='b')
                plt.title(f'Reward Progression Over Episodes (α={alpha}, γ={gamma}, ε={epsilon})')
                plt.xlabel('Episodes')
                plt.ylabel('Total Reward')
                plt.grid(True) 
                plt.legend()

                # Annotate last point
                last_episode = len(smoothed_rewards) - 1
                last_reward = smoothed_rewards[-1]

                # Annotate with offset for better visibility
                plt.annotate(f'{last_reward:.2f}', 
                             xy=(last_episode, last_reward), 
                             xytext=(last_episode - episodes * 0.02, last_reward + 0.2),
                             arrowprops=dict(arrowstyle='->', color='black'),
                             fontsize=10)

                # Save the plot
                image_filename = f"{output_folder}/alpha_{alpha}_gamma_{gamma}_epsilon_{epsilon}.png"
                plt.savefig(image_filename)
                plt.close()




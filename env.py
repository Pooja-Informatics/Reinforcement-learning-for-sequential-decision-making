

import numpy as np
class Environment:
    def __init__(self):
        self.rows = 3
        self.cols = 4
        self.actions = ['→', '↑', '←', '↓']
        self.state_rewards = {(2, 4): -1, (3, 4): 1}
        self.default_reward = -0.04
        self.state = (1, 1)  # Starting state

    # Resets the environment back to its initial condition
    def reset(self):
        self.state = (1, 1)  # Starting state
        return self.state

    # Takes a step in the environment
    def step(self, intended_action):
        # Define movement vectors for each action
        movements = {
            '→': (0, 1),   # Right
            '←': (0, -1),  # Left
            '↑': (-1, 0),  # Up
            '↓': (1, 0)    # Down
        }

        # Define the probabilities with 0.7 for the intended action and 0.1 for others
        probabilities = [0.1] * len(self.actions)
        intended_index = self.actions.index(intended_action)
        probabilities[intended_index] = 0.7

        # Choose an action based on these fixed probabilities
        chosen_action = np.random.choice(self.actions, p=probabilities)

        # Get the movement vector for the chosen action
        movement = movements.get(chosen_action)
        if movement is None:
            raise ValueError("Invalid action")

        # Calculate the next state
        next_state = (self.state[0] + movement[0], self.state[1] + movement[1])

        # Check if the next state is out of bounds
        if next_state[0] < 1 or next_state[0] > self.rows or next_state[1] < 1 or next_state[1] > self.cols:
            # If out of bounds, stay in the same state and penalize with default reward
            reward = self.default_reward
            done = False
        else:
            # Update the state to the next state
            self.state = next_state

            # Check if the next state has a specified reward
            if next_state in self.state_rewards:
                reward = self.state_rewards[next_state]
                done = True  # Episode ends if a goal state is reached
            else:
                reward = self.default_reward
                done = False  # Continue otherwise

        return self.state, reward, done



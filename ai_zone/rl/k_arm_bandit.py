import numpy as np
import random

class KArmBandit():
    def __init__(self, k):
        # Arrays to store true and estimated rewards
        self.true_reward = np.zeros(k)
        self.estimated_reward = np.zeros(k)
        self.actions_completed = np.zeros(k)
        
        # Parameters
        self.epsilon = 0.1
        self.k = k
        
        # Other trackers for evaluating an agent
        self.reward_history = []
        
        # Create initial true rewards
        for i in range(k):
            self.true_reward[i] = random.gauss(5, 1)
    
    def update_rewards(self):
        """
        Update true rewards after every step
        """
        for i in range(self.k):
            self.true_reward[i] += random.gauss(0, 0.05)
            
    def step(self):
        """
        Take an action, whether exploration or exploitation
        """
        # Determine the correct action to take
        if random.random() < self.epsilon:
            # Exploration
            action = random.randint(0, self.k - 1)
        else:
            # Exploitation
            action = np.argmax(self.agent_reward)
        
        # Get the reward
        reward = self.true_reward[action]
        self.reward_history.append(reward)
        
        # Update our estimates
        n = self.actions_completed[action]
        step_size = (1 / n) if n > 0 else 0
        self.estimated_reward[action] = reward + step_size * self.estimated_reward[action]
        self.actions_completed[action] += 1
        
        # Update the rewards
        self.update_rewards()
        
    def reset(self):
        # Arrays to store true and estimated rewards
        self.estimated_reward = np.zeros(self.k)
        self.actions_completed = np.zeros(self.k)
        
        # Other trackers for evaluating an agent
        self.reward_history = []
        
        
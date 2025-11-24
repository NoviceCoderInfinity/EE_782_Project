import numpy as np
import random
from collections import defaultdict
import json
import pickle


class QLearningAgent:
    """
    Tabular Q-Learning Agent for cloud task scheduling
    
    Uses a Q-table to map state-action pairs to Q-values
    State discretization for manageable table size
    """
    
    def __init__(
        self,
        num_vms=20,
        learning_rate=0.1,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        discretization_bins=3
    ):
        self.num_vms = num_vms
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.discretization_bins = discretization_bins
        
        # Q-table: dictionary mapping (state, action) -> Q-value
        self.q_table = defaultdict(lambda: np.zeros(num_vms))
        
        # Statistics
        self.update_count = 0
        self.state_visits = defaultdict(int)
        
    def discretize_state(self, state):
        """
        Discretize continuous state into bins for Q-table
        
        Args:
            state: Continuous state vector [vm_loads, next_cloudlet_length]
        
        Returns:
            Discretized state tuple (hashable)
        """
        discretized = []
        
        # Discretize VM loads (first num_vms elements)
        for i in range(self.num_vms):
            load = state[i]
            if load < 0.33:
                bin_val = 'LOW'
            elif load < 0.67:
                bin_val = 'MED'
            else:
                bin_val = 'HIGH'
            discretized.append(bin_val)
        
        # Discretize cloudlet length (last element)
        cloudlet_length = state[-1]
        if cloudlet_length < 0.33:
            discretized.append('SMALL')
        elif cloudlet_length < 0.67:
            discretized.append('MEDIUM')
        else:
            discretized.append('LARGE')
        
        return tuple(discretized)
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Continuous state vector
            training: Whether in training mode
        
        Returns:
            Selected action (VM index)
        """
        discrete_state = self.discretize_state(state)
        
        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            return random.randrange(self.num_vms)
        
        # Greedy: select action with highest Q-value
        q_values = self.q_table[discrete_state]
        return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-learning update rule
        
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Current Q-value
        current_q = self.q_table[discrete_state][action]
        
        # Max Q-value for next state
        if done:
            max_next_q = 0.0
        else:
            max_next_q = np.max(self.q_table[discrete_next_state])
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[discrete_state][action] = new_q
        
        # Update statistics
        self.update_count += 1
        self.state_visits[discrete_state] += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair"""
        discrete_state = self.discretize_state(state)
        return self.q_table[discrete_state][action]
    
    def get_state_value(self, state):
        """Get state value (max Q-value over actions)"""
        discrete_state = self.discretize_state(state)
        return np.max(self.q_table[discrete_state])
    
    def save(self, path):
        """Save Q-table and agent parameters"""
        save_dict = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'update_count': self.update_count,
            'state_visits': dict(self.state_visits),
            'num_vms': self.num_vms,
            'alpha': self.alpha,
            'gamma': self.gamma
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Q-table saved to {path}")
    
    def load(self, path):
        """Load Q-table and agent parameters"""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(self.num_vms), save_dict['q_table'])
        self.epsilon = save_dict['epsilon']
        self.update_count = save_dict['update_count']
        self.state_visits = defaultdict(int, save_dict['state_visits'])
        
        print(f"Q-table loaded from {path}")
        print(f"  States visited: {len(self.q_table)}")
        print(f"  Total updates: {self.update_count}")
    
    def get_statistics(self):
        """Get agent statistics"""
        return {
            'num_states': len(self.q_table),
            'total_updates': self.update_count,
            'epsilon': self.epsilon,
            'avg_visits_per_state': np.mean(list(self.state_visits.values())) if self.state_visits else 0
        }

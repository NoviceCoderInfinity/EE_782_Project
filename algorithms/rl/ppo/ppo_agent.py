"""
Proximal Policy Optimization (PPO) Agent

Actor-Critic architecture with clipped surrogate objective for stable policy learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class ActorNetwork(nn.Module):
    """Policy network (Actor)"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)


class CriticNetwork(nn.Module):
    """Value network (Critic)"""
    
    def __init__(self, state_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)


class PPOAgent:
    """
    Proximal Policy Optimization Agent for cloud task scheduling
    """
    
    def __init__(
        self,
        state_dim=21,
        action_dim=20,
        learning_rate=3e-4,
        gamma=0.99,
        epsilon_clip=0.2,
        epochs=10,
        batch_size=64,
        gae_lambda=0.95,
        value_coef=0.5,
        entropy_coef=0.01
    ):
        """
        Initialize PPO agent
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate for both networks
            gamma: Discount factor
            epsilon_clip: PPO clipping parameter
            epochs: Number of optimization epochs per update
            batch_size: Batch size for training
            gae_lambda: GAE lambda parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        # Statistics
        self.update_count = 0
    
    def select_action(self, state, training=True):
        """
        Select action using current policy
        
        Args:
            state: Current state
            training: Whether in training mode
        
        Returns:
            Selected action and log probability
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
        
        if training:
            # Sample from distribution
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            # Greedy selection
            action = torch.argmax(action_probs, dim=1)
            log_prob = torch.log(action_probs[0, action])
        
        return action.item(), log_prob.item(), value.item()
    
    def store_experience(self, state, action, reward, done, log_prob, value):
        """
        Store experience in buffer
        
        Args:
            state: State
            action: Action taken
            reward: Reward received
            done: Whether episode ended
            log_prob: Log probability of action
            value: State value estimate
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def compute_gae(self, next_value):
        """
        Compute Generalized Advantage Estimation
        
        Args:
            next_value: Value of final next state
        
        Returns:
            advantages: Advantage estimates
            returns: Return estimates
        """
        advantages = []
        gae = 0
        
        values = self.values + [next_value]
        
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                delta = self.rewards[t] - values[t]
                gae = delta
            else:
                delta = self.rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return advantages, returns
    
    def train_step(self, next_state):
        """
        Perform PPO update
        
        Args:
            next_state: Final next state for GAE computation
        
        Returns:
            Dictionary with loss values
        """
        if len(self.states) == 0:
            return None
        
        # Compute next value
        with torch.no_grad():
            next_value = self.critic(torch.FloatTensor(next_state).unsqueeze(0)).item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(self.states))
        actions_tensor = torch.LongTensor(self.actions)
        old_log_probs_tensor = torch.FloatTensor(self.log_probs)
        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO optimization epochs
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for epoch in range(self.epochs):
            # Generate random indices for mini-batches
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(self.states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(self.states))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Compute current policy
                action_probs = self.actor(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute ratio and clipped surrogate
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * batch_advantages
                
                # Actor loss
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                # Critic loss
                values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(values, batch_returns)
                
                # Update networks
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
        
        self.update_count += 1
        
        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        
        num_batches = (len(indices) // self.batch_size + 1) * self.epochs
        
        return {
            'actor_loss': total_actor_loss / num_batches,
            'critic_loss': total_critic_loss / num_batches,
            'entropy': total_entropy / num_batches
        }
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'update_count': self.update_count
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.update_count = checkpoint['update_count']
        print(f"Model loaded from {path}")

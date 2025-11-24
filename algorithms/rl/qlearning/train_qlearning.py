import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from utils.cloudsim_env import CloudSimEnv
from algorithms.rl.qlearning.qlearning_agent import QLearningAgent
import numpy as np
import json
from datetime import datetime


def train_qlearning(
    num_episodes=500,
    learning_rate=0.1,
    gamma=0.9,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    discretization_bins=3,
    log_interval=10,
    save_interval=50,
    results_dir='results/qlearning'
):
    """
    Train Q-Learning agent for cloud task scheduling
    
    Args:
        num_episodes: Number of training episodes
        learning_rate: Learning rate (alpha)
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Minimum exploration rate
        epsilon_decay: Epsilon decay rate
        discretization_bins: Number of bins for state discretization
        log_interval: Episodes between logging
        save_interval: Episodes between model saves
        results_dir: Directory to save results
    """
    # Create results directories
    os.makedirs(f'{results_dir}/logs', exist_ok=True)
    os.makedirs(f'{results_dir}/models', exist_ok=True)
    
    # Initialize environment and agent
    print("Initializing CloudSim environment...")
    env = CloudSimEnv()
    
    print("Creating Q-Learning agent...")
    agent = QLearningAgent(
        num_vms=20,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        discretization_bins=discretization_bins
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    training_log = []
    
    print(f"\nStarting Q-Learning training for {num_episodes} episodes...")
    print("=" * 60)
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Select and execute action
            action = agent.select_action(state, training=True)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update Q-table
            agent.update(state, action, reward, next_state, done or truncated)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Log progress
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_length = np.mean(episode_lengths[-log_interval:])
            stats = agent.get_statistics()
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  States: {stats['num_states']}")
            print(f"  Updates: {stats['total_updates']}")
            print("-" * 60)
            
            training_log.append({
                'episode': episode + 1,
                'avg_reward': float(avg_reward),
                'avg_length': float(avg_length),
                'epsilon': float(agent.epsilon),
                'num_states': stats['num_states'],
                'total_updates': stats['total_updates']
            })
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_path = f'{results_dir}/models/qlearning_ep{episode + 1}.pkl'
            agent.save(checkpoint_path)
    
    # Save final model
    final_model_path = f'{results_dir}/models/qlearning_final.pkl'
    agent.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Save training log
    log_path = f'{results_dir}/logs/training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(log_path, 'w') as f:
        json.dump({
            'hyperparameters': {
                'num_episodes': num_episodes,
                'learning_rate': learning_rate,
                'gamma': gamma,
                'epsilon_start': epsilon_start,
                'epsilon_end': epsilon_end,
                'epsilon_decay': epsilon_decay,
                'discretization_bins': discretization_bins
            },
            'training_log': training_log,
            'final_statistics': agent.get_statistics()
        }, f, indent=2)
    print(f"Training log saved to {log_path}")
    
    # Close environment
    env.close()
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Total episodes: {num_episodes}")
    print(f"States explored: {agent.get_statistics()['num_states']}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Q-Learning agent')
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Initial epsilon')
    parser.add_argument('--epsilon-end', type=float, default=0.01, help='Final epsilon')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay')
    parser.add_argument('--bins', type=int, default=3, help='Discretization bins')
    parser.add_argument('--results-dir', type=str, default='results/qlearning', help='Results directory')
    
    args = parser.parse_args()
    
    train_qlearning(
        num_episodes=args.episodes,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        discretization_bins=args.bins,
        results_dir=args.results_dir
    )

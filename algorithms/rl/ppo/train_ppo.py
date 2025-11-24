import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from utils.cloudsim_env import CloudSimEnv
from algorithms.rl.ppo.ppo_agent import PPOAgent
import numpy as np
import json
from datetime import datetime


def train_ppo(
    num_episodes=500,
    learning_rate=3e-4,
    gamma=0.99,
    epsilon_clip=0.2,
    epochs=10,
    batch_size=64,
    update_interval=2048,
    log_interval=10,
    save_interval=50,
    results_dir='results/ppo'
):
    """Train PPO agent for cloud task scheduling"""
    
    # Create results directories
    os.makedirs(f'{results_dir}/logs', exist_ok=True)
    os.makedirs(f'{results_dir}/models', exist_ok=True)
    
    # Initialize environment and agent
    print("Initializing CloudSim environment...")
    env = CloudSimEnv()
    
    print("Creating PPO agent...")
    agent = PPOAgent(
        state_dim=21,
        action_dim=20,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_clip=epsilon_clip,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    training_log = []
    
    print(f"\nStarting PPO training for {num_episodes} episodes...")
    print("=" * 60)
    
    global_step = 0
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Select action
            action, log_prob, value = agent.select_action(state, training=True)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, done or truncated, log_prob, value)
            
            episode_reward += reward
            episode_length += 1
            global_step += 1
            state = next_state
            
            # Update policy
            if global_step % update_interval == 0:
                losses = agent.train_step(next_state)
            
            if done or truncated:
                break
        
        # Update at end of episode if not updated
        if len(agent.states) > 0:
            losses = agent.train_step(state)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Log progress
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_length = np.mean(episode_lengths[-log_interval:])
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.2f}")
            if losses:
                print(f"  Actor Loss: {losses['actor_loss']:.4f}")
                print(f"  Critic Loss: {losses['critic_loss']:.4f}")
            print("-" * 60)
            
            training_log.append({
                'episode': episode + 1,
                'avg_reward': float(avg_reward),
                'avg_length': float(avg_length),
                'actor_loss': float(losses['actor_loss']) if losses else 0,
                'critic_loss': float(losses['critic_loss']) if losses else 0
            })
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_path = f'{results_dir}/models/ppo_ep{episode + 1}.pth'
            agent.save(checkpoint_path)
    
    # Save final model
    final_model_path = f'{results_dir}/models/ppo_final.pth'
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
                'epsilon_clip': epsilon_clip,
                'epochs': epochs,
                'batch_size': batch_size
            },
            'training_log': training_log
        }, f, indent=2)
    print(f"Training log saved to {log_path}")
    
    # Close environment
    env.close()
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO agent')
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--clip', type=float, default=0.2, help='PPO clip parameter')
    parser.add_argument('--epochs', type=int, default=10, help='Optimization epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--results-dir', type=str, default='results/ppo', help='Results directory')
    
    args = parser.parse_args()
    
    train_ppo(
        num_episodes=args.episodes,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_clip=args.clip,
        epochs=args.epochs,
        batch_size=args.batch_size,
        results_dir=args.results_dir
    )

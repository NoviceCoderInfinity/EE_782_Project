import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
import torch
from utils.cloudsim_env import CloudSimEnv
from algorithms.rl.dqn.dqn_agent import DQNAgent
import json
import time
from datetime import datetime


def train_dqn(
    num_episodes=500,
    max_steps_per_episode=100,
    save_freq=50,
    log_freq=10,
    model_save_path=None
):
    """
    Train DQN agent on CloudSim environment
    """
    
    # Create environment
    print("Initializing CloudSim environment...")
    env = CloudSimEnv(num_vms=20)
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=10
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    training_log = []
    
    # Model save path
    if model_save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = f"../../../results/models/dqn_cloudsim_{timestamp}.pth"
    
    print("\n" + "="*60)
    print("Starting DQN Training")
    print("="*60)
    
    start_time = time.time()
    
    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_loss = []
            
            for step in range(max_steps_per_episode):
                # Select action
                action = agent.select_action(state, training=True)
                
                # Execute action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store experience
                agent.store_experience(state, action, reward, next_state, done)
                
                # Train agent
                loss = agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Log episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step + 1)
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            episode_losses.append(avg_loss)
            
            # Log to console
            if (episode + 1) % log_freq == 0:
                avg_reward = np.mean(episode_rewards[-log_freq:])
                avg_length = np.mean(episode_lengths[-log_freq:])
                avg_loss_recent = np.mean(episode_losses[-log_freq:])
                
                print(f"Episode {episode+1}/{num_episodes}")
                print(f"  Avg Reward: {avg_reward:.3f}")
                print(f"  Avg Length: {avg_length:.1f}")
                print(f"  Avg Loss: {avg_loss_recent:.4f}")
                print(f"  Epsilon: {agent.epsilon:.3f}")
                print(f"  Buffer Size: {len(agent.replay_buffer)}")
                
                training_log.append({
                    'episode': episode + 1,
                    'avg_reward': float(avg_reward),
                    'avg_length': float(avg_length),
                    'avg_loss': float(avg_loss_recent),
                    'epsilon': float(agent.epsilon)
                })
            
            # Save model
            if (episode + 1) % save_freq == 0:
                agent.save(model_save_path)
                print(f"  Model saved at episode {episode+1}")
        
        # Final save
        agent.save(model_save_path)
        
        # Save training log
        log_path = model_save_path.replace('.pth', '_log.json')
        with open(log_path, 'w') as f:
            json.dump({
                'training_log': training_log,
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'episode_losses': episode_losses,
                'total_episodes': num_episodes,
                'total_time': time.time() - start_time
            }, f, indent=2)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
        print(f"Final avg reward (last 50 episodes): {np.mean(episode_rewards[-50:]):.3f}")
        print(f"Model saved to: {model_save_path}")
        print(f"Log saved to: {log_path}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save(model_save_path.replace('.pth', '_interrupted.pth'))
        
    finally:
        env.close()
    
    return agent, episode_rewards, episode_lengths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN agent on CloudSim')
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--save-freq', type=int, default=50, help='Model save frequency')
    parser.add_argument('--log-freq', type=int, default=10, help='Logging frequency')
    parser.add_argument('--model-path', type=str, default=None, help='Model save path')
    
    args = parser.parse_args()
    
    train_dqn(
        num_episodes=args.episodes,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        model_save_path=args.model_path
    )

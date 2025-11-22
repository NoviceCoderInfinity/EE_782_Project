"""
DQN Agent Training Script
Phase 3: Deep Q-Network for CloudSim Resource Management

This script trains a DQN agent to learn optimal VM selection policies
for cloud task scheduling using the CloudSim Gymnasium environment.

Features:
- Experience replay buffer
- Target network for stability
- Epsilon-greedy exploration
- Training metrics and logging
- Model checkpointing
"""

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloudsim_gym_env import CloudSimEnv


def create_dqn_agent(env, tensorboard_log="./dqn_cloudsim_tensorboard/"):
    """
    Create and configure DQN agent.
    
    Args:
        env: CloudSim Gymnasium environment
        tensorboard_log: Directory for TensorBoard logs
        
    Returns:
        DQN model
    """
    print("\n" + "="*70)
    print("DQN AGENT CONFIGURATION")
    print("="*70)
    
    # DQN Hyperparameters
    config = {
        'learning_rate': 1e-4,
        'buffer_size': 50000,
        'learning_starts': 1000,
        'batch_size': 64,
        'tau': 0.005,  # Soft update coefficient
        'gamma': 0.99,  # Discount factor
        'train_freq': 4,
        'gradient_steps': 1,
        'target_update_interval': 1000,
        'exploration_fraction': 0.3,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.05,
        'max_grad_norm': 10,
        'verbose': 1
    }
    
    print("\nHyperparameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create DQN model
    model = DQN(
        policy='MlpPolicy',
        env=env,
        learning_rate=config['learning_rate'],
        buffer_size=config['buffer_size'],
        learning_starts=config['learning_starts'],
        batch_size=config['batch_size'],
        tau=config['tau'],
        gamma=config['gamma'],
        train_freq=config['train_freq'],
        gradient_steps=config['gradient_steps'],
        target_update_interval=config['target_update_interval'],
        exploration_fraction=config['exploration_fraction'],
        exploration_initial_eps=config['exploration_initial_eps'],
        exploration_final_eps=config['exploration_final_eps'],
        max_grad_norm=config['max_grad_norm'],
        tensorboard_log=tensorboard_log,
        verbose=config['verbose'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    device = 'GPU' if torch.cuda.is_available() else 'CPU'
    print(f"\n✓ DQN agent created (using {device})")
    print(f"  Policy network: MLP")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    return model


def train_dqn(
    num_vms=30,
    total_timesteps=100000,
    save_freq=10000,
    eval_freq=5000,
    model_dir="./models/dqn/",
    log_dir="./logs/dqn/"
):
    """
    Train DQN agent on CloudSim environment.
    
    Args:
        num_vms: Number of VMs in simulation
        total_timesteps: Total training timesteps
        save_freq: Checkpoint save frequency
        eval_freq: Evaluation frequency
        model_dir: Directory to save models
        log_dir: Directory for logs
    """
    print("\n" + "="*70)
    print("DQN TRAINING - CLOUDSIM RESOURCE MANAGEMENT")
    print("="*70)
    print(f"\nTraining Configuration:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Number of VMs: {num_vms}")
    print(f"  Save frequency: {save_freq}")
    print(f"  Eval frequency: {eval_freq}")
    print(f"  Model directory: {model_dir}")
    print(f"  Log directory: {log_dir}")
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Create environment
        print("\n⏳ Creating CloudSim environment...")
        env = CloudSimEnv(num_vms=num_vms, host='localhost', port=5555)
        env = Monitor(env, log_dir)
        env = DummyVecEnv([lambda: env])
        
        print("✓ Environment created and wrapped")
        
        # Create DQN agent
        model = create_dqn_agent(env, tensorboard_log=log_dir)
        
        # Create callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=model_dir,
            name_prefix=f'dqn_cloudsim_{timestamp}'
        )
        
        eval_env = CloudSimEnv(num_vms=num_vms, host='localhost', port=5556)
        eval_env = Monitor(eval_env)
        eval_env = DummyVecEnv([lambda: eval_env])
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        callbacks = [checkpoint_callback, eval_callback]
        
        # Train the agent
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        print("\n⏳ Training DQN agent...")
        print("  (This may take a while. Monitor progress in TensorBoard)")
        print(f"  Run: tensorboard --logdir={log_dir}\n")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=100,
            progress_bar=True
        )
        
        # Save final model
        final_model_path = os.path.join(model_dir, f'dqn_cloudsim_final_{timestamp}.zip')
        model.save(final_model_path)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"\n✓ Training completed successfully!")
        print(f"✓ Final model saved: {final_model_path}")
        print(f"✓ Checkpoints saved in: {model_dir}")
        print(f"✓ Logs saved in: {log_dir}")
        
        return model, final_model_path
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print("  Saving current model...")
        interrupted_path = os.path.join(model_dir, f'dqn_cloudsim_interrupted_{timestamp}.zip')
        model.save(interrupted_path)
        print(f"✓ Model saved: {interrupted_path}")
        return model, interrupted_path
        
    except Exception as e:
        print(f"\n\n✗ Training error: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    finally:
        env.close()
        print("\n✓ Environment closed")


def evaluate_dqn(model_path, num_episodes=10, num_vms=30):
    """
    Evaluate trained DQN model.
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of evaluation episodes
        num_vms: Number of VMs
    """
    print("\n" + "="*70)
    print("DQN MODEL EVALUATION")
    print("="*70)
    print(f"\nModel: {model_path}")
    print(f"Episodes: {num_episodes}")
    
    try:
        # Load model
        print("\n⏳ Loading model...")
        model = DQN.load(model_path)
        print("✓ Model loaded")
        
        # Create environment
        print("⏳ Creating environment...")
        env = CloudSimEnv(num_vms=num_vms, host='localhost', port=5555)
        print("✓ Environment created")
        
        # Evaluate
        episode_rewards = []
        episode_lengths = []
        
        print(f"\n⏳ Running {num_episodes} evaluation episodes...\n")
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"  Episode {episode+1}/{num_episodes}: "
                  f"Reward={episode_reward:.2f}, Length={episode_length}")
        
        # Print statistics
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"\nEpisode Rewards:")
        print(f"  Mean: {np.mean(episode_rewards):.2f}")
        print(f"  Std: {np.std(episode_rewards):.2f}")
        print(f"  Min: {np.min(episode_rewards):.2f}")
        print(f"  Max: {np.max(episode_rewards):.2f}")
        print(f"\nEpisode Lengths:")
        print(f"  Mean: {np.mean(episode_lengths):.1f}")
        print(f"  Std: {np.std(episode_lengths):.1f}")
        
        env.close()
        print("\n✓ Evaluation complete")
        
        return episode_rewards, episode_lengths
        
    except Exception as e:
        print(f"\n✗ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN agent for CloudSim')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                       help='Mode: train or eval')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--vms', type=int, default=30,
                       help='Number of VMs')
    parser.add_argument('--model', type=str, default=None,
                       help='Model path for evaluation')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("\n🚀 Starting DQN Training...")
        model, model_path = train_dqn(
            num_vms=args.vms,
            total_timesteps=args.timesteps
        )
        
        if model_path:
            print(f"\n✓ Training successful!")
            print(f"  Model saved at: {model_path}")
            print(f"\nTo evaluate: python train_dqn.py --mode eval --model {model_path}")
    
    elif args.mode == 'eval':
        if not args.model:
            print("✗ Error: --model path required for evaluation")
            return
        
        print("\n🔍 Starting DQN Evaluation...")
        evaluate_dqn(args.model, num_episodes=args.eval_episodes, num_vms=args.vms)


if __name__ == "__main__":
    main()

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
import torch
from utils.cloudsim_env import CloudSimEnv
from algorithms.rl.dqn.dqn_agent import DQNAgent
import json
import time


def evaluate_dqn(
    model_path,
    num_episodes=10,
    render=False
):
    """
    Evaluate trained DQN agent
    """
    
    # Create environment
    print("Initializing CloudSim environment for evaluation...")
    env = CloudSimEnv(num_vms=20)
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim
    )
    
    # Load model
    agent.load(model_path)
    agent.epsilon = 0.0  # Greedy policy for evaluation
    
    print("\n" + "="*60)
    print("Evaluating DQN Agent")
    print("="*60)
    
    episode_rewards = []
    episode_lengths = []
    episode_infos = []
    
    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Select action (greedy)
                action = agent.select_action(state, training=False)
                
                # Execute action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    episode_infos.append(info)
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"\nEpisode {episode+1}/{num_episodes}")
            print(f"  Reward: {episode_reward:.3f}")
            print(f"  Length: {episode_length}")
            if 'avg_response_time' in info:
                print(f"  Avg Response Time: {info.get('avg_response_time', 0):.3f}")
                print(f"  SLA Violations: {info.get('sla_violations', 0)}")
                print(f"  SLA Violation Rate: {info.get('sla_violation_rate', 0):.3%}")
        
        # Print summary
        print("\n" + "="*60)
        print("Evaluation Summary")
        print("="*60)
        print(f"Average Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
        print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        
        if episode_infos:
            avg_response_times = [info.get('avg_response_time', 0) for info in episode_infos]
            sla_violations = [info.get('sla_violations', 0) for info in episode_infos]
            
            print(f"Average Response Time: {np.mean(avg_response_times):.3f} ± {np.std(avg_response_times):.3f}")
            print(f"Average SLA Violations: {np.mean(sla_violations):.1f} ± {np.std(sla_violations):.1f}")
        
        # Save results
        results_path = model_path.replace('.pth', '_eval_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'episode_infos': episode_infos,
                'summary': {
                    'mean_reward': float(np.mean(episode_rewards)),
                    'std_reward': float(np.std(episode_rewards)),
                    'mean_length': float(np.mean(episode_lengths)),
                    'std_length': float(np.std(episode_lengths))
                }
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
        print("="*60)
        
    finally:
        env.close()
    
    return episode_rewards, episode_lengths, episode_infos


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate DQN agent on CloudSim')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    evaluate_dqn(
        model_path=args.model_path,
        num_episodes=args.episodes
    )

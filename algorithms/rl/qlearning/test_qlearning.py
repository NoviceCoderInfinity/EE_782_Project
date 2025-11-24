import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from utils.cloudsim_env import CloudSimEnv
from algorithms.rl.qlearning.qlearning_agent import QLearningAgent
import numpy as np
import json
from datetime import datetime


def evaluate_qlearning(
    model_path,
    num_episodes=100,
    results_dir='results/qlearning'
):
    """
    Evaluate trained Q-Learning agent
    
    Args:
        model_path: Path to trained model
        num_episodes: Number of evaluation episodes
        results_dir: Directory to save results
    """
    # Create results directory
    os.makedirs(f'{results_dir}/logs', exist_ok=True)
    
    # Initialize environment and agent
    print("Initializing CloudSim environment...")
    env = CloudSimEnv()
    
    print(f"Loading Q-Learning agent from {model_path}...")
    agent = QLearningAgent(num_vms=20)
    agent.load(model_path)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    sla_violations = []
    
    print(f"\nEvaluating for {num_episodes} episodes...")
    print("=" * 60)
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Greedy action selection (no exploration)
            action = agent.select_action(state, training=False)
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        sla_violations.append(info.get('sla_violations', 0))
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed")
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    avg_sla = np.mean(sla_violations)
    
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Average Episode Length: {avg_length:.2f}")
    print(f"  Average SLA Violations: {avg_sla:.2f}")
    print(f"  Min Reward: {np.min(episode_rewards):.2f}")
    print(f"  Max Reward: {np.max(episode_rewards):.2f}")
    print("=" * 60)
    
    # Save results
    results = {
        'model_path': model_path,
        'num_episodes': num_episodes,
        'avg_reward': float(avg_reward),
        'std_reward': float(std_reward),
        'avg_length': float(avg_length),
        'avg_sla_violations': float(avg_sla),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'all_rewards': [float(r) for r in episode_rewards],
        'all_lengths': [float(l) for l in episode_lengths],
        'all_sla_violations': [float(s) for s in sla_violations],
        'agent_statistics': agent.get_statistics()
    }
    
    results_path = f'{results_dir}/logs/evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Close environment
    env.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Q-Learning agent')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--results-dir', type=str, default='results/qlearning', help='Results directory')
    
    args = parser.parse_args()
    
    evaluate_qlearning(
        model_path=args.model,
        num_episodes=args.episodes,
        results_dir=args.results_dir
    )

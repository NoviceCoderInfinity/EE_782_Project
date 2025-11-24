"""
Visualization utilities for training results
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_training_results(log_path, save_dir=None):
    """
    Plot training results from log file
    
    Args:
        log_path: Path to training log JSON file
        save_dir: Directory to save plots (if None, displays plots)
    """
    with open(log_path, 'r') as f:
        data = json.load(f)
    
    training_log = data.get('training_log', [])
    episode_rewards = data.get('episode_rewards', [])
    episode_losses = data.get('episode_losses', [])
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN Training Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Moving average
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 
                color='red', linewidth=2, label=f'Moving Avg ({window})')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss
    ax2 = axes[0, 1]
    ax2.plot(episode_losses, alpha=0.5, color='orange')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Epsilon decay
    if training_log:
        episodes = [entry['episode'] for entry in training_log]
        epsilons = [entry['epsilon'] for entry in training_log]
        
        ax3 = axes[1, 0]
        ax3.plot(episodes, epsilons, color='green', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.set_title('Epsilon Decay')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Average reward per training log
    if training_log:
        episodes = [entry['episode'] for entry in training_log]
        avg_rewards = [entry['avg_reward'] for entry in training_log]
        
        ax4 = axes[1, 1]
        ax4.plot(episodes, avg_rewards, color='purple', linewidth=2, marker='o')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Average Reward')
        ax4.set_title('Average Reward (per log interval)')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'training_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_evaluation_results(eval_path, save_dir=None):
    """
    Plot evaluation results
    
    Args:
        eval_path: Path to evaluation results JSON file
        save_dir: Directory to save plots
    """
    with open(eval_path, 'r') as f:
        data = json.load(f)
    
    episode_rewards = data.get('episode_rewards', [])
    episode_infos = data.get('episode_infos', [])
    
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('DQN Evaluation Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Rewards distribution
    ax1 = axes[0]
    ax1.bar(range(len(episode_rewards)), episode_rewards, color='steelblue', alpha=0.7)
    ax1.axhline(y=np.mean(episode_rewards), color='red', linestyle='--', 
                label=f'Mean: {np.mean(episode_rewards):.2f}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Performance metrics
    if episode_infos:
        metrics = {
            'Avg Response Time': [info.get('avg_response_time', 0) for info in episode_infos],
            'SLA Violations': [info.get('sla_violations', 0) for info in episode_infos]
        }
        
        ax2 = axes[1]
        x = np.arange(len(episode_infos))
        width = 0.35
        
        ax2_twin = ax2.twinx()
        
        bars1 = ax2.bar(x - width/2, metrics['Avg Response Time'], width, 
                       label='Avg Response Time', color='coral', alpha=0.7)
        bars2 = ax2_twin.bar(x + width/2, metrics['SLA Violations'], width,
                            label='SLA Violations', color='lightgreen', alpha=0.7)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Avg Response Time', color='coral')
        ax2_twin.set_ylabel('SLA Violations', color='green')
        ax2.set_title('Performance Metrics')
        ax2.tick_params(axis='y', labelcolor='coral')
        ax2_twin.tick_params(axis='y', labelcolor='green')
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'evaluation_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize training/evaluation results')
    parser.add_argument('--log-path', type=str, help='Path to training log JSON')
    parser.add_argument('--eval-path', type=str, help='Path to evaluation results JSON')
    parser.add_argument('--save-dir', type=str, default='../results/plots', 
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    if args.log_path:
        plot_training_results(args.log_path, args.save_dir)
    
    if args.eval_path:
        plot_evaluation_results(args.eval_path, args.save_dir)

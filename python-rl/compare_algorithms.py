"""
Model Comparison and Visualization
Phase 3: Compare Q-Learning, DQN, and PPO performance

This script compares the performance of different RL algorithms:
- Q-Learning (baseline from Phase 1)
- DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)

Generates comparison plots and performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(results_dir='./results'):
    """
    Load training results for all algorithms.
    
    Args:
        results_dir: Directory containing result files
        
    Returns:
        Dictionary with results for each algorithm
    """
    results = {}
    results_path = Path(results_dir)
    
    for algo in ['qlearning', 'dqn', 'ppo']:
        result_file = results_path / f'{algo}_results.json'
        if result_file.exists():
            with open(result_file, 'r') as f:
                results[algo] = json.load(f)
        else:
            print(f"⚠ Warning: {result_file} not found")
    
    return results


def plot_training_curves(results, save_path='./plots/training_curves.png'):
    """
    Plot training reward curves for all algorithms.
    """
    plt.figure(figsize=(12, 6))
    
    for algo, data in results.items():
        if 'episode_rewards' in data:
            rewards = data['episode_rewards']
            episodes = range(len(rewards))
            
            # Plot raw rewards
            plt.plot(episodes, rewards, alpha=0.3, label=f'{algo.upper()} (raw)')
            
            # Plot moving average
            window = min(100, len(rewards) // 10)
            if window > 1:
                ma = pd.Series(rewards).rolling(window=window).mean()
                plt.plot(episodes, ma, linewidth=2, label=f'{algo.upper()} (MA-{window})')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Curves: Q-Learning vs DQN vs PPO')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✓ Saved training curves: {save_path}")
    plt.close()


def plot_performance_comparison(results, save_path='./plots/performance_comparison.png'):
    """
    Bar plot comparing final performance metrics.
    """
    metrics = ['avg_reward', 'avg_response_time', 'avg_throughput', 'sla_violations']
    algorithms = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        values = []
        labels = []
        
        for algo in algorithms:
            if metric in results[algo]:
                values.append(results[algo][metric])
                labels.append(algo.upper())
        
        if values:
            bars = axes[idx].bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(labels)])
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}',
                             ha='center', va='bottom')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✓ Saved performance comparison: {save_path}")
    plt.close()


def plot_resource_utilization(results, save_path='./plots/resource_utilization.png'):
    """
    Plot resource utilization comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    resources = ['cpu_util', 'ram_util', 'load_balance']
    titles = ['CPU Utilization', 'RAM Utilization', 'Load Balance']
    
    for idx, (resource, title) in enumerate(zip(resources, titles)):
        data = []
        labels = []
        
        for algo, result in results.items():
            if resource in result:
                data.append(result[resource])
                labels.append(algo.upper())
        
        if data:
            axes[idx].bar(labels, data, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(labels)])
            axes[idx].set_ylabel('Percentage' if 'util' in resource else 'Score')
            axes[idx].set_title(title)
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, (label, value) in enumerate(zip(labels, data)):
                axes[idx].text(i, value, f'{value:.1f}%' if 'util' in resource else f'{value:.2f}',
                             ha='center', va='bottom')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✓ Saved resource utilization: {save_path}")
    plt.close()


def generate_summary_table(results, save_path='./results/summary_table.csv'):
    """
    Generate summary table of all metrics.
    """
    summary_data = []
    
    for algo, data in results.items():
        row = {'Algorithm': algo.upper()}
        row.update(data)
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Reorder columns for better readability
    priority_cols = ['Algorithm', 'avg_reward', 'avg_response_time', 'avg_throughput', 
                     'cpu_util', 'load_balance', 'sla_violations']
    remaining_cols = [c for c in df.columns if c not in priority_cols]
    df = df[priority_cols + remaining_cols]
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\n✓ Saved summary table: {save_path}")
    
    # Print table
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return df


def find_best_algorithm(results):
    """
    Determine which algorithm performed best based on weighted metrics.
    """
    print("\n" + "="*80)
    print("BEST ALGORITHM ANALYSIS")
    print("="*80)
    
    # Define weights for different metrics (higher is better)
    weights = {
        'avg_reward': 0.3,
        'avg_response_time': -0.2,  # Negative because lower is better
        'avg_throughput': 0.2,
        'cpu_util': 0.1,
        'load_balance': -0.1,  # Negative because lower imbalance is better
        'sla_violations': -0.1  # Negative because fewer is better
    }
    
    scores = {}
    for algo, data in results.items():
        score = 0
        for metric, weight in weights.items():
            if metric in data:
                # Normalize metrics to 0-1 range (approximately)
                value = data[metric]
                if 'time' in metric:
                    value = 100 / max(value, 1)  # Invert time metrics
                elif 'violations' in metric:
                    value = max(0, 10 - value)  # Fewer violations is better
                
                score += weight * value
        
        scores[algo] = score
    
    best_algo = max(scores, key=scores.get)
    
    print("\nWeighted Scores:")
    for algo, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {algo.upper()}: {score:.4f}")
    
    print(f"\n🏆 Best Algorithm: {best_algo.upper()}")
    print(f"   Score: {scores[best_algo]:.4f}")
    print("="*80)
    
    return best_algo


def main():
    """Main comparison script."""
    print("\n" + "="*80)
    print("RL ALGORITHMS COMPARISON - PHASE 3")
    print("="*80)
    
    # Load results
    print("\n⏳ Loading results...")
    results = load_results('./results')
    
    if not results:
        print("\n✗ No results found!")
        print("  Run training scripts first to generate results:")
        print("    - Q-Learning: CloudSimQLearningSimulation.java")
        print("    - DQN: python train_dqn.py --mode train")
        print("    - PPO: python train_ppo.py --mode train")
        return
    
    print(f"✓ Loaded results for: {', '.join([a.upper() for a in results.keys()])}")
    
    # Generate visualizations
    print("\n⏳ Generating visualizations...")
    
    if len([r for r in results.values() if 'episode_rewards' in r]) > 0:
        plot_training_curves(results)
    
    plot_performance_comparison(results)
    plot_resource_utilization(results)
    
    # Generate summary table
    generate_summary_table(results)
    
    # Find best algorithm
    find_best_algorithm(results)
    
    print("\n✓ Comparison complete!")
    print("\nGenerated files:")
    print("  - plots/training_curves.png")
    print("  - plots/performance_comparison.png")
    print("  - plots/resource_utilization.png")
    print("  - results/summary_table.csv")


if __name__ == "__main__":
    main()

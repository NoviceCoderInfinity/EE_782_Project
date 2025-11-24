# Q-Learning for Cloud Task Scheduling

Tabular Q-Learning implementation for cloud task scheduling using CloudSim Plus.

## Algorithm Overview

Q-Learning is a model-free reinforcement learning algorithm that learns the value of actions in states. It uses a Q-table to store state-action values and updates them using the Bellman equation.

### Key Features

- **Tabular representation**: Q-table stores value for each state-action pair
- **State discretization**: Continuous states binned into discrete categories (LOW/MED/HIGH)
- **Epsilon-greedy exploration**: Balance between exploration and exploitation
- **Off-policy learning**: Learns optimal policy while following epsilon-greedy policy

## Files

- `qlearning_agent.py`: Q-Learning agent implementation with Q-table
- `train_qlearning.py`: Training script for Q-Learning agent
- `test_qlearning.py`: Evaluation script for trained Q-Learning agent

## Training

```bash
python algorithms/rl/qlearning/train_qlearning.py --episodes 500 --lr 0.1 --gamma 0.9
```

### Hyperparameters

- `--episodes`: Number of training episodes (default: 500)
- `--lr`: Learning rate alpha (default: 0.1)
- `--gamma`: Discount factor (default: 0.9)
- `--epsilon-start`: Initial exploration rate (default: 1.0)
- `--epsilon-end`: Final exploration rate (default: 0.01)
- `--epsilon-decay`: Epsilon decay rate per episode (default: 0.995)
- `--bins`: Number of discretization bins (default: 3)

## Evaluation

```bash
python algorithms/rl/qlearning/test_qlearning.py --model results/qlearning/models/qlearning_final.pkl --episodes 100
```

## State Discretization

The agent discretizes continuous state values into bins:

- **VM Load**: LOW (<0.33), MED (0.33-0.67), HIGH (>0.67)
- **Cloudlet Length**: SMALL (<0.33), MEDIUM (0.33-0.67), LARGE (>0.67)

With 20 VMs and 1 cloudlet dimension, this creates manageable state space.

## Q-Learning Update Rule

```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Where:
- α: Learning rate
- γ: Discount factor
- r: Immediate reward
- s, s': Current and next states
- a, a': Current and next actions

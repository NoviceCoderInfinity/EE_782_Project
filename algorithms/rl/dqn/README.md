# DQN for CloudSim Task Scheduling

Deep Q-Network (DQN) implementation for intelligent task scheduling in cloud environments using CloudSim Plus.

## Architecture

- **DQN Agent** (`dqn_agent.py`): Deep Q-Network with experience replay and target network
- **Training** (`train_dqn.py`): Training script with logging and model checkpointing
- **Evaluation** (`test_dqn.py`): Evaluation script for trained models

## Features

- Experience replay buffer for sample efficiency
- Target network for stable training
- Epsilon-greedy exploration strategy
- Multi-objective reward (response time + load balancing)
- Model checkpointing and training logs
- Comprehensive evaluation metrics

## Quick Start

### Prerequisites

Make sure CloudSim server is running:
```bash
cd ../../../simulation/java
mvn compile exec:java
```

### Training

```bash
# Basic training (500 episodes)
python train_dqn.py

# Custom training
python train_dqn.py --episodes 1000 --save-freq 100 --log-freq 20
```

### Evaluation

```bash
python test_dqn.py --model-path ../../../results/models/dqn_cloudsim_TIMESTAMP.pth --episodes 10
```

## Hyperparameters

- Learning rate: 1e-3
- Discount factor (gamma): 0.99
- Epsilon decay: 0.995 (1.0 → 0.01)
- Replay buffer size: 10,000
- Batch size: 64
- Target network update frequency: 10 steps

## Output

Training produces:
- Model checkpoint: `results/models/dqn_cloudsim_TIMESTAMP.pth`
- Training log: `results/models/dqn_cloudsim_TIMESTAMP_log.json`
- Evaluation results: `results/models/dqn_cloudsim_TIMESTAMP_eval_results.json`

# Python RL Agents - Phase 3

## Deep Reinforcement Learning for CloudSim Resource Management

This directory contains Python implementations of advanced RL algorithms (DQN and PPO) that communicate with CloudSim Plus via socket communication.

## 📁 Directory Structure

```
python-rl/
├── cloudsim_gym_env.py      # Gymnasium environment wrapper for CloudSim
├── train_dqn.py              # DQN agent training script
├── train_ppo.py              # PPO agent training script
├── compare_algorithms.py     # Performance comparison and visualization
├── models/                   # Saved model checkpoints
│   ├── dqn/
│   └── ppo/
├── logs/                     # Training logs and TensorBoard data
│   ├── dqn/
│   └── ppo/
├── results/                  # Evaluation results and metrics
└── plots/                    # Generated comparison plots
```

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have the Python environment set up:

```bash
conda activate cloudsim_rl

# Verify packages
pip list | grep -E "gymnasium|stable-baselines3|torch|matplotlib"
```

If packages are missing:

```bash
pip install gymnasium stable-baselines3 torch matplotlib pandas numpy
```

### 2. Start Java CloudSim Server

First, start the Java CloudSim simulation server with RLBroker:

```bash
cd ../cloudsim-rl-project

# Using Maven (if working):
mvn exec:java -Dexec.mainClass="com.ee782.cloudsim.CloudSimRLSimulation"

# Or using IDE:
# Run CloudSimRLSimulation.java in IntelliJ/Eclipse
```

The Java server will listen on port 5555 for Python client connections.

### 3. Train DQN Agent

```bash
python train_dqn.py --mode train --timesteps 100000 --vms 30
```

**Parameters:**

- `--mode`: `train` or `eval`
- `--timesteps`: Total training timesteps (default: 100000)
- `--vms`: Number of VMs in simulation (default: 30)
- `--model`: Model path (for evaluation mode)

**Monitoring Training:**

```bash
tensorboard --logdir=./logs/dqn/
```

Then open http://localhost:6006 in your browser.

### 4. Train PPO Agent

```bash
python train_ppo.py --mode train --timesteps 100000 --vms 30
```

**Monitoring Training:**

```bash
tensorboard --logdir=./logs/ppo/
```

### 5. Evaluate Trained Models

**Evaluate DQN:**

```bash
python train_dqn.py --mode eval --model ./models/dqn/dqn_cloudsim_final_20251121_120000.zip --eval-episodes 10
```

**Evaluate PPO:**

```bash
python train_ppo.py --mode eval --model ./models/ppo/ppo_cloudsim_final_20251121_120000.zip --eval-episodes 10
```

### 6. Compare Algorithms

After training all algorithms (Q-Learning, DQN, PPO), compare their performance:

```bash
python compare_algorithms.py
```

This generates:

- Training curves plot
- Performance comparison bar charts
- Resource utilization comparison
- Summary table (CSV)

## 📊 CloudSim Gymnasium Environment

### Environment Specification

**`cloudsim_gym_env.py`** wraps CloudSim as a Gymnasium environment.

#### Observation Space

```python
Box(0, 1, shape=(num_vms * 3,), dtype=float32)
```

Flattened array containing for each VM:

- CPU utilization (0-1)
- RAM utilization (0-1)
- Bandwidth utilization (0-1)

**Example:** For 30 VMs, observation shape is `(90,)`

#### Action Space

```python
Discrete(num_vms)
```

Select VM ID (0 to num_vms-1) for task placement.

#### Reward Function

Multi-objective reward combining:

```python
reward = w_time * time_reward + w_balance * balance_reward + w_sla * sla_penalty
```

Where:

- **time_reward**: -response_time / 100.0 (negative, minimize)
- **balance_reward**: 1.0 - load_imbalance (positive, maximize)
- **sla_penalty**: -sla_violations \* 10.0 (negative, minimize)

Default weights:

- `w_time = 0.5`
- `w_balance = 0.3`
- `w_sla = 0.2`

#### Communication Protocol

JSON-based socket communication with Java:

**Python → Java (Commands):**

```json
// Reset environment
{"command": "reset", "episode": 1}

// Take action (select VM)
{"command": "step", "action": 5, "step": 10}
```

**Java → Python (Responses):**

```json
{
  "status": "success",
  "vm_states": [
    {"cpu": 0.75, "ram": 0.60, "bw": 0.40},
    ...
  ],
  "done": false,
  "cloudlets_completed": 50,
  "response_time": 12.5,
  "waiting_time": 3.2,
  "load_imbalance": 0.15,
  "sla_violation": 0
}
```

## 🤖 Algorithms

### DQN (Deep Q-Network)

**Architecture:**

- MLP policy network: Observation → Q-values for all actions
- Target network for stability
- Experience replay buffer (50,000 transitions)

**Key Hyperparameters:**

```python
learning_rate = 1e-4
buffer_size = 50000
batch_size = 64
gamma = 0.99
target_update_interval = 1000
exploration_initial_eps = 1.0
exploration_final_eps = 0.05
```

**Advantages:**

- Sample efficient (reuses past experiences)
- Proven for discrete action spaces
- Stable with target networks

**When to use:**

- When sample efficiency is critical
- Discrete action spaces
- Off-policy learning desired

### PPO (Proximal Policy Optimization)

**Architecture:**

- Actor-Critic with MLP networks
- Policy network: Observation → Action probabilities
- Value network: Observation → State value

**Key Hyperparameters:**

```python
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.01
```

**Advantages:**

- More stable than vanilla policy gradients
- Works well for both continuous and discrete actions
- Less sensitive to hyperparameters

**When to use:**

- When training stability is important
- On-policy learning preferred
- Good default choice for many problems

## 📈 Training Tips

### 1. Start Small

Begin with fewer timesteps to verify the setup:

```bash
python train_dqn.py --mode train --timesteps 10000
```

### 2. Monitor TensorBoard

Watch key metrics during training:

- Episode reward (should increase over time)
- Episode length (varies by workload)
- Loss values (should decrease/stabilize)
- Exploration epsilon (DQN: should decay)

### 3. Adjust Reward Weights

Modify reward weights in `cloudsim_gym_env.py`:

```python
reward_weights = {
    'time': 0.6,      # Prioritize response time
    'balance': 0.2,   # Less emphasis on load balance
    'sla': 0.2        # SLA violations
}
```

### 4. Hyperparameter Tuning

Key hyperparameters to tune:

**DQN:**

- `learning_rate`: Try 1e-3, 1e-4, 1e-5
- `buffer_size`: Increase for more diverse experiences
- `exploration_fraction`: Adjust exploration schedule

**PPO:**

- `learning_rate`: Try 1e-3, 3e-4, 1e-4
- `n_steps`: Increase for more stable updates
- `clip_range`: Try 0.1, 0.2, 0.3

### 5. Dealing with Unstable Training

If training is unstable:

**DQN:**

- Increase `learning_starts` (more random exploration first)
- Decrease `learning_rate`
- Increase `target_update_interval`

**PPO:**

- Decrease `learning_rate`
- Increase `n_steps` (collect more data before update)
- Decrease `clip_range` (more conservative updates)

## 🐛 Troubleshooting

### Connection Refused Error

```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Solution:** Ensure Java CloudSim server is running and listening on port 5555.

### Environment Reset Timeout

```
socket.timeout: timed out
```

**Solution:**

- Check Java server logs for errors
- Increase timeout in `cloudsim_gym_env.py`:

```python
self.socket.settimeout(60.0)  # Increase from 30s to 60s
```

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:** Force CPU training:

```python
# In train_dqn.py or train_ppo.py
device='cpu'
```

Or reduce batch size:

```python
batch_size = 32  # Reduce from 64
```

### Training Too Slow

**Solutions:**

1. Reduce number of VMs (fewer states)
2. Use GPU if available
3. Reduce `n_steps` or `buffer_size`
4. Train on simpler workload first

## 📊 Expected Results

After training (100K timesteps):

### DQN

- Episode Reward: ~150-250 (depends on reward scaling)
- Response Time: 10-15 seconds average
- Load Balance: 70-80% balanced
- Training Time: ~2-4 hours (CPU)

### PPO

- Episode Reward: ~200-300 (typically higher than DQN)
- Response Time: 8-12 seconds average
- Load Balance: 75-85% balanced
- Training Time: ~1-3 hours (CPU)

### Q-Learning (Baseline)

- Episode Reward: ~100-150
- Response Time: 15-20 seconds average
- Load Balance: 60-70% balanced

## 🎯 Next Steps

1. **Experiment with different reward functions**
2. **Try different workloads** (modify synthetic_workload.csv)
3. **Implement multi-objective optimization**
4. **Add more sophisticated state features**
5. **Compare with heuristic baselines** (Phase 4)

## 📚 References

- **Stable Baselines3:** https://stable-baselines3.readthedocs.io/
- **Gymnasium:** https://gymnasium.farama.org/
- **DQN Paper:** Mnih et al., "Human-level control through deep reinforcement learning" (2015)
- **PPO Paper:** Schulman et al., "Proximal Policy Optimization Algorithms" (2017)

---

**Phase 3 Status:** Implementation Complete ✅

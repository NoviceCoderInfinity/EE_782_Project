# Phase 3 Quick Start Guide

## Running DQN and PPO Agents

This guide provides quick commands to get Phase 3 running.

## 🚀 Quick Start (5 Steps)

### Step 1: Verify Environment

```bash
conda activate cloudsim_rl

# Check Python packages
python -c "import gymnasium, stable_baselines3, torch; print('✓ All packages installed')"
```

### Step 2: Start Java CloudSim Server

**Option A: Using IDE (Recommended)**

```
1. Open IntelliJ IDEA or Eclipse
2. Navigate to: cloudsim-rl-project/src/main/java/com/ee782/cloudsim/
3. Right-click CloudSimRLSimulation.java
4. Select "Run"
```

**Option B: Using Maven**

```bash
cd cloudsim-rl-project
mvn exec:java -Dexec.mainClass="com.ee782.cloudsim.CloudSimRLSimulation"
```

**Expected Output:**

```
✓ Socket server started on port 5555
⏳ Waiting for Python client connection...
```

### Step 3: Train DQN (New Terminal)

```bash
cd python-rl
python train_dqn.py --mode train --timesteps 10000
```

**For full training (100K timesteps):**

```bash
python train_dqn.py --mode train --timesteps 100000
```

### Step 4: Train PPO (After DQN)

```bash
python train_ppo.py --mode train --timesteps 10000
```

### Step 5: Compare Results

```bash
python compare_algorithms.py
```

## 📊 Monitoring Training

**Open TensorBoard:**

```bash
# DQN monitoring
tensorboard --logdir=./logs/dqn/

# PPO monitoring
tensorboard --logdir=./logs/ppo/
```

Then open: http://localhost:6006

## 🎯 Key Commands Reference

### Training

```bash
# DQN - Quick test (10K steps, ~10 minutes)
python train_dqn.py --mode train --timesteps 10000 --vms 30

# DQN - Full training (100K steps, ~2-4 hours)
python train_dqn.py --mode train --timesteps 100000 --vms 30

# PPO - Quick test
python train_ppo.py --mode train --timesteps 10000 --vms 30

# PPO - Full training
python train_ppo.py --mode train --timesteps 100000 --vms 30
```

### Evaluation

```bash
# Evaluate DQN model
python train_dqn.py --mode eval \
    --model ./models/dqn/dqn_cloudsim_final_TIMESTAMP.zip \
    --eval-episodes 10

# Evaluate PPO model
python train_ppo.py --mode eval \
    --model ./models/ppo/ppo_cloudsim_final_TIMESTAMP.zip \
    --eval-episodes 10
```

### Testing Environment

```bash
# Test CloudSim Gym environment
python -c "from cloudsim_gym_env import test_environment; test_environment()"
```

## 📁 Directory Structure

```
python-rl/
├── cloudsim_gym_env.py          # Gymnasium environment
├── train_dqn.py                 # DQN training script
├── train_ppo.py                 # PPO training script
├── compare_algorithms.py        # Comparison tool
├── README.md                    # Full documentation
│
├── models/                      # Saved models
│   ├── dqn/
│   │   ├── dqn_cloudsim_final_TIMESTAMP.zip
│   │   └── dqn_cloudsim_checkpoint_*.zip
│   └── ppo/
│       ├── ppo_cloudsim_final_TIMESTAMP.zip
│       └── ppo_cloudsim_checkpoint_*.zip
│
├── logs/                        # TensorBoard logs
│   ├── dqn/
│   └── ppo/
│
├── results/                     # Evaluation results
│   ├── qlearning_results.json
│   ├── dqn_results.json
│   └── ppo_results.json
│
└── plots/                       # Generated plots
    ├── training_curves.png
    ├── performance_comparison.png
    └── resource_utilization.png
```

## 🐛 Troubleshooting

### Problem 1: Connection Refused

```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Solution:** Java server not running. Start `CloudSimRLSimulation.java` first.

### Problem 2: Port Already in Use

```
OSError: [Errno 98] Address already in use
```

**Solution:** Kill existing process:

```bash
lsof -ti:5555 | xargs kill -9
```

### Problem 3: CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:** Force CPU training (edit script):

```python
device='cpu'
```

### Problem 4: Import Errors

```
ModuleNotFoundError: No module named 'gymnasium'
```

**Solution:** Install missing packages:

```bash
conda activate cloudsim_rl
pip install gymnasium stable-baselines3 torch matplotlib pandas
```

## 📈 Expected Training Times

**CPU (Intel i7/AMD Ryzen):**

- 10K timesteps: ~10-15 minutes
- 100K timesteps: ~2-4 hours

**GPU (NVIDIA):**

- 10K timesteps: ~5-8 minutes
- 100K timesteps: ~1-2 hours

## ✅ Success Indicators

**Java Server:**

```
✓ Socket server started on port 5555
⏳ Waiting for Python client connection...
✓ Python client connected from 127.0.0.1
```

**Python Training:**

```
✓ DQN agent created (using CPU/GPU)
✓ Environment created and wrapped
⏳ Training DQN agent...
[Progress bar showing episodes and rewards]
```

**TensorBoard:**

- `rollout/ep_rew_mean` should increase over time
- `train/loss` should decrease and stabilize
- No NaN or Inf values

## 🎯 Quick Comparison

After training both agents, generate comparison:

```bash
python compare_algorithms.py
```

**Outputs:**

- `plots/training_curves.png` - Learning curves
- `plots/performance_comparison.png` - Metrics comparison
- `plots/resource_utilization.png` - Resource usage
- `results/summary_table.csv` - Numerical results

## 📚 More Information

- **Full Phase 3 Documentation:** `PHASE_3_COMPLETION_SUMMARY.md`
- **Detailed Python Guide:** `python-rl/README.md`
- **Algorithm Details:** See training script docstrings
- **Hyperparameter Tuning:** Modify config dicts in training scripts

## 🎓 Learning Resources

- **Stable Baselines3 Docs:** https://stable-baselines3.readthedocs.io/
- **Gymnasium Tutorial:** https://gymnasium.farama.org/tutorials/
- **DQN Paper:** Mnih et al. (2015) - Nature
- **PPO Paper:** Schulman et al. (2017) - arXiv

---

**Phase 3 Status:** ✅ Complete and Ready to Run!

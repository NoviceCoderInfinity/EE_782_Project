# 📚 Documentation Index

Quick navigation to all project documentation.

---

## 🚀 Getting Started

1. **[README.md](../README.md)** - Start here!
   - Project overview
   - Architecture diagrams
   - Quick setup guide
   - Usage instructions

2. **[SETUP_GUIDE.md](../SETUP_GUIDE.md)** - Detailed installation
   - Prerequisites
   - Step-by-step setup
   - Environment configuration
   - Troubleshooting

3. **[PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md)** - What's been done
   - Completed tasks
   - File structure
   - Demo guide
   - Status overview

---

## 📖 Technical Documentation

### Algorithm Documentation

- **[ALGORITHMS.md](ALGORITHMS.md)** - Deep dive into algorithms
  - DQN implementation details
  - PPO (planned)
  - Q-Learning (planned)
  - Heuristic algorithms (planned)
  - Comparative analysis

### API Reference

- **[API.md](API.md)** - Complete API documentation
  - Python API (CloudSimEnv, DQNAgent)
  - Java API (CloudSimSocketServer)
  - Training scripts
  - Visualization tools
  - Configuration

---

## 💻 Code Documentation

### DQN Implementation

- **[algorithms/rl/dqn/README.md](../algorithms/rl/dqn/README.md)**
  - DQN-specific documentation
  - Architecture
  - Features
  - Quick start
  - Hyperparameters

### Source Code

Key files to review:

1. **DQN Agent**: `algorithms/rl/dqn/dqn_agent.py`
   - Network architecture
   - Experience replay
   - Training logic

2. **CloudSim Server**: `simulation/java/src/main/java/org/ee782/CloudSimSocketServer.java`
   - Socket server
   - Simulation logic
   - State/reward calculation

3. **Gym Environment**: `utils/cloudsim_env.py`
   - Environment wrapper
   - Action/observation spaces
   - Communication protocol

4. **Training**: `algorithms/rl/dqn/train_dqn.py`
   - Training loop
   - Logging
   - Model checkpointing

5. **Evaluation**: `algorithms/rl/dqn/test_dqn.py`
   - Testing logic
   - Metrics calculation

6. **Visualization**: `utils/visualization.py`
   - Plot generation
   - Result analysis

---

## 🛠️ Setup & Configuration

### Installation Scripts

- **[verify_setup.sh](../verify_setup.sh)** - Verify your installation
- **[quick_start.sh](../quick_start.sh)** - Automated training

### Configuration Files

- **[requirements.txt](../requirements.txt)** - Python dependencies
- **[pom.xml](../simulation/java/pom.xml)** - Maven configuration

---

## 📊 Results & Outputs

### Directory Structure

```
results/
├── logs/        # Training logs
├── models/      # Saved models (.pth files)
└── plots/       # Generated visualizations
```

### Output Files

After training:
- `dqn_cloudsim_TIMESTAMP.pth` - Model checkpoint
- `dqn_cloudsim_TIMESTAMP_log.json` - Training log
- `dqn_cloudsim_TIMESTAMP_eval_results.json` - Evaluation results
- `training_results.png` - Training curves
- `evaluation_results.png` - Performance metrics

---

## 📝 Planning Documents

- **[to_do.md](../to_do.md)** - Original project plan
  - Phase-by-phase breakdown
  - Timeline
  - Detailed requirements

---

## 🎯 Quick Reference

### For First-Time Setup
1. Read [README.md](../README.md)
2. Follow [SETUP_GUIDE.md](../SETUP_GUIDE.md)
3. Run `./verify_setup.sh`
4. Try `./quick_start.sh`

### For Understanding Algorithms
1. Read [ALGORITHMS.md](ALGORITHMS.md)
2. Review `algorithms/rl/dqn/dqn_agent.py`
3. Check `algorithms/rl/dqn/README.md`

### For API Reference
1. Read [API.md](API.md)
2. Check inline code comments
3. Review training/test scripts

### For Demo/Presentation
1. Read [PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md)
2. Review [README.md](../README.md) architecture section
3. Check [ALGORITHMS.md](ALGORITHMS.md) for technical details

---

## 🔗 External Resources

### CloudSim Plus
- GitHub: https://github.com/cloudsimplus/cloudsimplus
- Docs: https://cloudsimplus.org/

### Deep Learning
- PyTorch: https://pytorch.org/
- Gymnasium: https://gymnasium.farama.org/

### Datasets
- Google Cluster Trace: https://github.com/google/cluster-data
- Alibaba Cluster Trace: https://github.com/alibaba/clusterdata

---

## 📞 Support

For questions or issues:
- Check [SETUP_GUIDE.md](../SETUP_GUIDE.md) troubleshooting section
- Review [API.md](API.md) error handling section
- Check inline code documentation

---

**Last Updated**: November 24, 2025

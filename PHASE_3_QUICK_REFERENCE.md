# Phase 3 Quick Reference Card

## 🚀 Start Training (2 Commands)

### Terminal 1 - Java Server

```bash
cd ~/Desktop/EE_782_Project/cloudsim-rl-project
mvn exec:java -Dexec.mainClass="com.ee782.cloudsim.CloudSimRLSimulation"
```

### Terminal 2 - Python Training

```bash
cd ~/Desktop/EE_782_Project/python-rl
conda activate cloudsim_rl
python train_dqn.py      # For DQN
# OR
python train_ppo.py      # For PPO
```

---

## 📊 Monitor Training

### TensorBoard

```bash
cd ~/Desktop/EE_782_Project/python-rl
tensorboard --logdir=./tensorboard_logs
```

Open: http://localhost:6006

### Watch Logs

```bash
tail -f /tmp/cloudsim_server.log  # Java server logs
```

---

## 🔍 Check Status

### Is Java Server Running?

```bash
netstat -tuln | grep 5555
ps aux | grep CloudSimRLSimulation
```

### Is Python Environment Active?

```bash
conda env list | grep cloudsim_rl
```

### GPU Available?

```bash
nvidia-smi
```

---

## 🛠️ If Something Goes Wrong

### Restart Everything

```bash
# Kill any existing Java processes
pkill -f CloudSimRLSimulation

# Recompile Java
cd ~/Desktop/EE_782_Project/cloudsim-rl-project
mvn clean compile

# Start fresh
mvn exec:java -Dexec.mainClass="com.ee782.cloudsim.CloudSimRLSimulation"
```

### Reinstall Python Dependencies

```bash
conda activate cloudsim_rl
pip install --upgrade stable-baselines3 tensorboard torch gymnasium
```

---

## 📁 Key Files

### Java (cloudsim-rl-project/)

- `CloudSimRLSimulation.java` - Main simulation entry point
- `RLBroker.java` - Socket server, handles Python communication
- `pom.xml` - Maven config (CloudSim Plus 8.0.0)

### Python (python-rl/)

- `cloudsim_gym_env.py` - Gymnasium environment wrapper
- `train_dqn.py` - DQN training script
- `train_ppo.py` - PPO training script
- `compare_algorithms.py` - Compare trained models

### Outputs

- `models/dqn/` - Saved DQN models
- `models/ppo/` - Saved PPO models
- `logs/` - Training logs
- `tensorboard_logs/` - TensorBoard data
- `results/` - Comparison plots and metrics

---

## 🎯 Expected Training Time

- **DQN (100K steps)**: ~30-45 minutes (with GPU)
- **PPO (100K steps)**: ~25-35 minutes (with GPU)
- **Evaluation**: ~5 minutes per model

---

## 📈 Success Indicators

✅ Java: "Waiting for Python client connection..."
✅ Python: "Connected to CloudSim server at localhost:5555"
✅ Python: "Expected 30 VM states, got 30" (no warning)
✅ TensorBoard: Reward curve increasing
✅ Models: Files saved in models/ directory

---

## 🆘 Quick Fixes

| Problem              | Solution                                     |
| -------------------- | -------------------------------------------- |
| Port 5555 in use     | `sudo lsof -ti:5555 \| xargs kill -9`        |
| Maven errors         | `mvn clean compile`                          |
| Python import errors | `conda activate cloudsim_rl`                 |
| GPU not detected     | Check CUDA: `nvcc --version`                 |
| TensorBoard 404      | Delete `tensorboard_logs/`, restart training |
| Out of memory        | Reduce batch*size in train*\*.py             |

---

## 🔗 Useful Commands

```bash
# View all Maven phases
mvn help:describe -Dcmd=compile

# Python package info
conda list | grep stable

# Check CloudSim version
grep -A 2 "cloudsimplus" cloudsim-rl-project/pom.xml

# Clean all outputs
rm -rf python-rl/{models,logs,tensorboard_logs,results}/*

# Git status (if tracking)
git status
git diff
```

---

## 📞 Documentation References

- **CloudSim Plus**: https://cloudsimplus.org/
- **Stable Baselines3**: https://stable-baselines3.readthedocs.io/
- **Gymnasium**: https://gymnasium.farama.org/
- **TensorBoard**: https://www.tensorflow.org/tensorboard

---

**Last Updated**: Nov 21, 2025  
**Status**: ✅ Fully Operational  
**Next Step**: Phase 4 - Heuristic Baselines

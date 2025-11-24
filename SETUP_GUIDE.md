# EE 782 Project: Cloud Task Scheduling with Deep Reinforcement Learning

A comprehensive implementation of intelligent task scheduling algorithms for cloud computing environments using CloudSim Plus and Deep Reinforcement Learning.

## 📁 Repository Structure

```
EE_782_Project/
├── algorithms/
│   ├── rl/                      # Reinforcement Learning algorithms
│   │   ├── dqn/                 # Deep Q-Network
│   │   ├── ppo/                 # Proximal Policy Optimization (TODO)
│   │   └── qlearning/           # Q-Learning (TODO)
│   ├── ml/                      # Machine Learning algorithms (TODO)
│   └── heuristic/               # Heuristic algorithms (TODO)
├── simulation/
│   ├── java/                    # CloudSim Plus Java simulation
│   └── configs/                 # Simulation configurations
├── results/
│   ├── logs/                    # Training logs
│   ├── models/                  # Saved models
│   └── plots/                   # Visualizations
├── utils/                       # Utility modules
│   ├── cloudsim_env.py         # Gym environment wrapper
│   └── visualization.py         # Plotting utilities
└── docs/                        # Documentation
```

## 🚀 Quick Start

### Prerequisites

**Java Environment:**
- JDK 17 or higher
- Maven 3.6+

**Python Environment:**
- Python 3.9+
- Conda (recommended)

### Installation

1. **Clone the repository:**
```bash
cd /home/anupam/win_desktop/EE_782_Project
```

2. **Set up Python environment:**
```bash
conda create -n ee782 python=3.9
conda activate ee782
pip install -r requirements.txt
```

3. **Build CloudSim Java simulation:**
```bash
cd simulation/java
mvn clean compile
```

### Running DQN Training

**Step 1: Start CloudSim Server**

In one terminal:
```bash
cd simulation/java
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
mvn exec:java -Dexec.mainClass="org.ee782.CloudSimSocketServer"
```

**Step 2: Train DQN Agent**

In another terminal:
```bash
cd algorithms/rl/dqn
python train_dqn.py --episodes 500 --save-freq 50 --log-freq 10
```

**Step 3: Evaluate Trained Model**

```bash
cd algorithms/rl/dqn
python test_dqn.py --model-path ../../../results/models/dqn_cloudsim_TIMESTAMP.pth --episodes 10
```

**Step 4: Visualize Results**

```bash
cd utils
python visualization.py --log-path ../results/models/dqn_cloudsim_TIMESTAMP_log.json --save-dir ../results/plots
```

## 🏗️ Architecture

### Python-Java Bridge

The system uses a socket-based communication architecture:

```
┌─────────────────┐         Socket         ┌──────────────────┐
│   Python Agent  │  ◄────────────────►   │  CloudSim Plus   │
│   (DQN/PPO)     │   JSON Protocol       │   (Java Server)  │
└─────────────────┘                        └──────────────────┘
```

**Protocol:**
- **Reset**: `{"command": "reset"}` → Returns initial state
- **Step**: `{"command": "step", "action": <vm_id>}` → Returns `{state, reward, done, info}`
- **Close**: `{"command": "close"}` → Closes connection

### DQN Implementation

**Key Features:**
- Experience Replay Buffer (10,000 transitions)
- Target Network (updates every 10 steps)
- Epsilon-greedy exploration (1.0 → 0.01)
- Multi-objective reward function

**Network Architecture:**
```
Input (state_dim) → FC(128) → ReLU → FC(128) → ReLU → Output (action_dim)
```

**State Space:** `[vm_load_1, ..., vm_load_n, next_cloudlet_length]`

**Action Space:** Discrete(20) - Select VM for task assignment

**Reward Function:**
```
R = 0.7 * (-estimated_response_time/100) + 0.3 * (-load_variance/1000)
```

## 📊 Metrics

The system tracks the following performance metrics:

- **Average Response Time**: Mean time from task submission to completion
- **Throughput**: Tasks completed per unit time
- **SLA Violations**: Percentage of tasks missing deadlines
- **Load Imbalance**: Standard deviation of VM loads
- **Episode Reward**: Cumulative reward per episode

## 🔧 Configuration

Edit `simulation/configs/` for custom configurations (TODO):
- Number of hosts, VMs, cloudlets
- VM capacities (MIPS, RAM, bandwidth)
- Workload patterns
- SLA requirements

## 📝 Next Steps

### Phase 2: Data Pipeline
- [ ] Trace data parser implementation
- [ ] Synthetic workload generator

### Phase 3: Additional RL Algorithms
- [ ] PPO implementation
- [ ] Q-Learning baseline
- [ ] A3C/A2C variants

### Phase 4: Heuristic Baselines
- [ ] Round Robin
- [ ] Genetic Algorithm
- [ ] Ant Colony Optimization

### Phase 5: Evaluation & Analysis
- [ ] Comparative study
- [ ] Statistical analysis
- [ ] Paper writing

## 📚 References

- CloudSim Plus: https://github.com/cloudsimplus/cloudsimplus
- Stable Baselines3: https://stable-baselines3.readthedocs.io/
- Gymnasium: https://gymnasium.farama.org/

## 📄 License

This project is for academic purposes (EE 782 Course Project).

## 👥 Contributors

- Anupam (NoviceCoderInfinity)

---

**Status**: Phase 0 & Phase 1 Complete ✅ | DQN Implementation Ready ✅

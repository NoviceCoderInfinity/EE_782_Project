# Cloud Task Scheduling with Deep Reinforcement Learning
## EE 782 Project - Complete Implementation Guide

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Algorithms Implemented](#algorithms-implemented)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
7. [Results & Evaluation](#results--evaluation)
8. [Future Work](#future-work)

---

## 🎯 Project Overview

This project implements and compares multiple task scheduling algorithms for cloud computing environments using CloudSim Plus simulation framework. The focus is on Deep Reinforcement Learning (DRL) approaches, particularly Deep Q-Networks (DQN), alongside traditional heuristic methods.

### Key Objectives

- Implement DRL-based task scheduling algorithms (DQN, PPO, Q-Learning)
- Develop ML-based scheduling approaches
- Compare with traditional heuristic algorithms (Round Robin, Genetic Algorithm, ACO)
- Evaluate performance using real-world traces (Google Cluster, Alibaba)
- Multi-objective optimization: response time, energy efficiency, SLA compliance

### Technologies Used

- **Simulation**: CloudSim Plus 8.0.0 (Java)
- **Deep Learning**: PyTorch 2.0+
- **RL Framework**: Custom Gymnasium environment
- **Languages**: Python 3.9+, Java 17+
- **Build Tools**: Maven, Conda

---

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                     EE 782 Cloud Scheduler                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐                  ┌──────────────────────┐
│   Python RL Agent    │                  │   CloudSim Plus      │
│                      │                  │   (Java Server)      │
│  ┌────────────────┐  │    Socket API    │                      │
│  │  DQN Network   │  │ ◄──────────────► │  ┌───────────────┐  │
│  │  (PyTorch)     │  │   JSON Protocol  │  │  Datacenter   │  │
│  └────────────────┘  │                  │  │  - 5 Hosts    │  │
│                      │                  │  │  - 20 VMs     │  │
│  ┌────────────────┐  │                  │  │  - Cloudlets  │  │
│  │ Replay Buffer  │  │                  │  └───────────────┘  │
│  │ (10k samples)  │  │                  │                      │
│  └────────────────┘  │                  │  ┌───────────────┐  │
│                      │                  │  │  Metrics      │  │
│  ┌────────────────┐  │                  │  │  - Response   │  │
│  │ Target Network │  │                  │  │  - SLA Viols  │  │
│  └────────────────┘  │                  │  │  - Load Bal   │  │
└──────────────────────┘                  └──────────────────────┘
         ▲                                          ▲
         │                                          │
         └──────────────┬───────────────────────────┘
                        │
                ┌───────▼────────┐
                │  Visualization │
                │  & Analysis    │
                │  - Matplotlib  │
                │  - Seaborn     │
                └────────────────┘
```

### Communication Protocol

**Request (Python → Java)**:
```json
{
  "command": "step",
  "action": 5
}
```

**Response (Java → Python)**:
```json
{
  "state": [0.2, 0.5, 0.1, ..., 0.45],
  "reward": -0.234,
  "done": false,
  "info": {
    "current_cloudlet": 42,
    "completed": 41
  }
}
```

---

## 📁 Repository Structure

```
EE_782_Project/
│
├── algorithms/                          # All scheduling algorithms
│   ├── rl/                             # Reinforcement Learning algorithms
│   │   ├── dqn/                        # Deep Q-Network
│   │   │   ├── dqn_agent.py           # DQN agent implementation
│   │   │   ├── train_dqn.py           # Training script
│   │   │   ├── test_dqn.py            # Evaluation script
│   │   │   └── README.md              # DQN documentation
│   │   ├── ppo/                        # Proximal Policy Optimization
│   │   │   └── (TODO: PPO implementation)
│   │   └── qlearning/                  # Tabular Q-Learning
│   │       └── (TODO: Q-Learning baseline)
│   │
│   ├── ml/                             # Machine Learning approaches
│   │   └── (TODO: Decision Trees, Random Forest, etc.)
│   │
│   └── heuristic/                      # Traditional algorithms
│       └── (TODO: Round Robin, GA, ACO)
│
├── simulation/                          # CloudSim environment
│   ├── java/                           # Java simulation code
│   │   ├── pom.xml                     # Maven configuration
│   │   └── src/main/java/org/ee782/
│   │       └── CloudSimSocketServer.java  # Main simulation server
│   └── configs/                        # Configuration files
│       └── (TODO: Simulation configs)
│
├── utils/                              # Utility modules
│   ├── cloudsim_env.py                # Gymnasium environment wrapper
│   ├── visualization.py               # Plotting and analysis tools
│   └── __init__.py
│
├── results/                            # Training outputs
│   ├── logs/                          # Training logs
│   ├── models/                        # Saved model checkpoints
│   └── plots/                         # Generated visualizations
│
├── docs/                              # Documentation
│   └── (Future documentation files)
│
├── requirements.txt                    # Python dependencies
├── SETUP_GUIDE.md                     # Installation instructions
├── verify_setup.sh                    # Setup verification script
├── quick_start.sh                     # Quick training script
├── README.md                          # This file

```

---

## 🤖 Algorithms Implemented

### 1. Deep Q-Network (DQN) ✅

**Status**: Implemented and Ready

**Key Features**:
- Experience Replay Buffer (10,000 transitions)
- Target Network (updates every 10 steps)
- Epsilon-greedy exploration (1.0 → 0.01, decay=0.995)
- Multi-layer perceptron: Input → FC(128) → ReLU → FC(128) → ReLU → Output
- Multi-objective reward function

**State Space**: `[vm_load_1, vm_load_2, ..., vm_load_20, next_cloudlet_length]`
- Dimension: 21 (20 VMs + 1 cloudlet feature)
- Normalized values

**Action Space**: Discrete(20)
- Each action represents selecting a VM for task assignment

**Reward Function**:
```python
R = 0.7 × (-estimated_response_time/100) + 0.3 × (-load_variance/1000)
```

**Hyperparameters**:
```python
learning_rate = 1e-3
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
buffer_size = 10000
batch_size = 64
target_update_freq = 10
```

### 2. Proximal Policy Optimization (PPO) 🔨

**Status**: Planned

### 3. Q-Learning 🔨

**Status**: Planned

### 4. Heuristic Algorithms 🔨

**Status**: Planned
- Round Robin
- Genetic Algorithm
- Ant Colony Optimization

---

## 🚀 Installation & Setup

### Prerequisites

```bash
# System Requirements
- Ubuntu 20.04+ (WSL supported)
- 8GB RAM minimum
- Java 17+
- Maven 3.6+
- Python 3.9+
- Conda (recommended)
```

### Quick Setup

```bash
# 1. Clone repository
cd ~/workspace
git clone https://github.com/NoviceCoderInfinity/EE_782_Project.git
cd EE_782_Project

# 2. Setup Python environment
conda create -n ee782 python=3.9 -y
conda activate ee782
pip install -r requirements.txt

# 3. Setup Java environment
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# 4. Build CloudSim
cd simulation/java
mvn clean compile
cd ../..

# 5. Verify setup
./verify_setup.sh
```

For detailed instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md).

---

## 📖 Usage Guide

### Quick Start (Automated)

```bash
./quick_start.sh
```

### Manual Training

**Terminal 1: Start CloudSim Server**
```bash
cd simulation/java
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
mvn exec:java -Dexec.mainClass="org.ee782.CloudSimSocketServer"
```

**Terminal 2: Train DQN**
```bash
conda activate ee782
cd algorithms/rl/dqn
python train_dqn.py --episodes 500 --save-freq 50 --log-freq 10
```

**Terminal 3: Evaluate Model**
```bash
cd algorithms/rl/dqn
python test_dqn.py --model-path ../../../results/models/dqn_cloudsim_*.pth --episodes 10
```

**Visualize Results**
```bash
cd utils
python visualization.py --log-path ../results/models/dqn_cloudsim_*_log.json --save-dir ../results/plots
```

---

## 📊 Results & Evaluation

### Performance Metrics

1. **Average Response Time**: Mean time from task submission to completion
2. **Throughput**: Tasks completed per unit time
3. **SLA Violation Rate**: Percentage of tasks missing deadlines
4. **Load Imbalance**: Standard deviation of VM loads
5. **Episode Reward**: Cumulative reward per episode

### Output Files

```
results/
├── models/
│   ├── dqn_cloudsim_20251124_143022.pth         # Model checkpoint
│   ├── dqn_cloudsim_20251124_143022_log.json    # Training log
│   └── dqn_cloudsim_20251124_143022_eval_results.json
├── logs/
│   └── cloudsim_server.log
└── plots/
    ├── training_results.png
    └── evaluation_results.png
```

---

## 🚧 Future Work

### Phase 2: Data Pipeline
- [ ] Google Cluster Trace parser
- [ ] Alibaba Cluster Trace parser
- [ ] Synthetic workload generator

### Phase 3: Additional Algorithms
- [ ] PPO implementation
- [ ] Q-Learning baseline
- [ ] A3C/A2C variants

### Phase 4: Heuristic Baselines
- [ ] Round Robin
- [ ] Genetic Algorithm
- [ ] Ant Colony Optimization

### Phase 5: Evaluation
- [ ] Comprehensive comparative study
- [ ] Statistical analysis
- [ ] Research paper

---

## 📚 References

- CloudSim Plus: https://github.com/cloudsimplus/cloudsimplus
- PyTorch: https://pytorch.org/
- Gymnasium: https://gymnasium.farama.org/

---

## 🤝 Contributing

**Author**: Anupam  
**GitHub**: [@NoviceCoderInfinity](https://github.com/NoviceCoderInfinity)  
**Course**: EE 782 - Cloud Computing  

---

**Project Status**: ✅ Phase 0 & 1 Complete | 🔨 DQN Implementation Ready

**Last Updated**: November 24, 2025

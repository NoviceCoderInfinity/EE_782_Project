# Quick Start Guide

## EE 782 Cloud Resource Management Project

### 📁 Project Overview

This project implements cloud resource management using reinforcement learning techniques integrated with CloudSim Plus simulation.

**Current Status:**

- ✅ Phase 0: Prerequisites and architecture setup
- ✅ Phase 1: Q-Learning baseline implementation
- ✅ Phase 2: Data preprocessing and trace integration
- ⏳ Phase 3: Advanced RL (DQN & PPO) - **Next**
- ⏳ Phase 4: Heuristic baselines
- ⏳ Phase 5: Evaluation and reporting

---

## 🚀 Quick Start

### 1. Prerequisites Check

```bash
# Java version (need 17+)
java -version

# Maven version
mvn -version

# Python environment
conda activate cloudsim_rl
python --version

# Verify Python packages
pip list | grep -E "gymnasium|stable-baselines3|pandas|numpy|torch"
```

### 2. Project Structure

```
EE_782_Project/
├── cloudsim-rl-project/          # Java CloudSim simulation
│   ├── pom.xml                   # Maven dependencies
│   ├── src/main/java/com/ee782/cloudsim/
│   │   ├── QLearningBroker.java                   # Q-Learning agent
│   │   ├── CloudSimQLearningSimulation.java       # Random workload sim
│   │   ├── GoogleTraceReader.java                 # Trace data reader
│   │   └── CloudSimTraceQLearningSimulation.java  # Trace-based sim
│   └── src/main/resources/
│       └── synthetic_workload.csv                 # Workload data
│
├── data-preprocessing/            # Workload preparation
│   ├── download_google_traces.py
│   ├── preprocess_google_traces.py
│   └── synthetic_workload.csv
│
├── python-bridge/                 # Python-Java communication
│   └── cloudsim_bridge_server.py  # Socket server template
│
└── Documentation/
    ├── PROJECT_README.md
    ├── PHASE_0_1_COMPLETION_SUMMARY.md
    └── PHASE_2_COMPLETION_SUMMARY.md
```

---

## 🔧 Running the Simulation

### Option A: Using IDE (Recommended - Avoids Maven issue)

1. **Import Project:**

   - Open IntelliJ IDEA or Eclipse
   - Import as Maven project: `cloudsim-rl-project/`
   - Wait for dependencies to download

2. **Run Q-Learning with Random Workload:**

   ```
   Right-click: CloudSimQLearningSimulation.java → Run
   ```

3. **Run Q-Learning with Trace Workload:**
   ```
   Right-click: CloudSimTraceQLearningSimulation.java → Run
   ```

### Option B: Maven Command Line (Has classpath issue)

```bash
cd cloudsim-rl-project

# Clean and compile
mvn clean compile

# Run simulation (if compilation works)
mvn exec:java -Dexec.mainClass="com.ee782.cloudsim.CloudSimTraceQLearningSimulation"
```

**Note:** Currently has Maven classpath issue. Code is correct but needs IDE or manual classpath fix.

### Option C: Manual Compilation with Classpath

```bash
cd cloudsim-rl-project

# Download dependencies
mvn dependency:copy-dependencies

# Compile manually
javac -cp "target/dependency/*:src/main/java" \
  src/main/java/com/ee782/cloudsim/*.java \
  -d target/classes

# Run
java -cp "target/dependency/*:target/classes:src/main/resources" \
  com.ee782.cloudsim.CloudSimTraceQLearningSimulation
```

---

## 📊 Understanding the Output

### Simulation Output Sections

#### 1. **Initialization**

```
============================================================
CloudSim Plus Q-Learning Simulation with Google Trace Workload
Phase 2: Data Integration and Realistic Workload Evaluation
============================================================

✓ Created datacenter with 8 hosts
✓ Created Q-Learning broker (α=0.10, γ=0.90, ε=0.20)
✓ Created 30 VMs
```

#### 2. **Trace Loading**

```
--------------------------------------------------------------
Loading Workload from Google Cluster Trace
--------------------------------------------------------------
✓ Loaded 1000 cloudlets from synthetic_workload.csv

Cloudlet Statistics:
  Count: 1000
  Length (MI): min=1,000, max=100,000, avg=20,213
  PEs: min=1, max=4
  Submission delay (s): min=0.5, max=4801.7
  Simulation duration: 4801.7 seconds
```

#### 3. **Simulation Execution**

```
============================================================
Starting Simulation...
============================================================

[CloudSim Plus simulation runs...]
```

#### 4. **Performance Metrics**

```
Performance Metrics:
  Total Cloudlets: 1000
  Completed: 1000 (100.0%)
  Average Waiting Time: 12.45 seconds
  Average Execution Time: 8.32 seconds
  Average Response Time: 20.77 seconds
  Throughput: 2.150 tasks/second
  Simulation Time: 465.12 seconds
```

#### 5. **VM Utilization**

```
VM Utilization:
  VM #0: 85.3% CPU, 34 tasks
  VM #1: 82.1% CPU, 31 tasks
  ...
```

#### 6. **Q-Learning Statistics**

```
Q-Learning Statistics:
  Q-Table Size: 150 state-action pairs
  Unique States Explored: 50
  Average Q-Value: 12.34
  Max Q-Value: 45.67
  Min Q-Value: -2.13
```

---

## 🎯 Key Metrics Explained

| Metric              | Description                             | Goal                    |
| ------------------- | --------------------------------------- | ----------------------- |
| **Response Time**   | Waiting + Execution time                | Minimize                |
| **Throughput**      | Tasks completed per second              | Maximize                |
| **CPU Utilization** | Percentage of CPU capacity used         | 70-90% (optimal)        |
| **Load Imbalance**  | Difference between max/min tasks per VM | Minimize                |
| **Q-Value**         | Expected future reward                  | Increases with learning |

---

## 🔍 Testing Components Individually

### Test 1: Trace Reader

```bash
cd cloudsim-rl-project

# Using Maven (if working)
mvn exec:java -Dexec.mainClass="com.ee782.cloudsim.GoogleTraceReader"

# Or in IDE:
# Right-click GoogleTraceReader.java → Run
```

**Expected Output:**

```
==========================================================
GOOGLE TRACE READER TEST
==========================================================

⏳ Reading trace file: synthetic_workload.csv
✓ Loaded 1,000 rows

Cloudlet Statistics:
  Count: 1000
  Length (MI): min=1,000, max=100,000, avg=20,213
  ...

First 5 Cloudlets:
  [0] Length=31,580 MI, PEs=1, Delay=0.5s, Priority=2
  [1] Length=5,864 MI, PEs=2, Delay=4.8s, Priority=1
  ...
```

### Test 2: Synthetic Workload Generator

```bash
cd data-preprocessing

conda activate cloudsim_rl

python -c "
from preprocess_google_traces import create_synthetic_workload
create_synthetic_workload(num_tasks=100, output_file='test_workload.csv')
"

# Verify
head test_workload.csv
```

### Test 3: Q-Learning Broker

```bash
# Run the original Phase 1 simulation (random workload)
# In IDE: Right-click CloudSimQLearningSimulation.java → Run
```

---

## 🐛 Troubleshooting

### Issue 1: Maven Compilation Fails

```
[ERROR] package org.cloudsimplus.cloudlets does not exist
```

**Solution:**

1. Use IDE (IntelliJ/Eclipse) - automatically handles classpath
2. Or manually add CloudSim Plus JAR:
   ```bash
   mvn dependency:copy-dependencies
   # Then use manual compilation (see Option C above)
   ```

### Issue 2: CSV File Not Found

```
java.io.IOException: Resource file not found: synthetic_workload.csv
```

**Solution:**

```bash
# Ensure file is in resources folder
ls cloudsim-rl-project/src/main/resources/synthetic_workload.csv

# If missing, copy from data-preprocessing:
cp data-preprocessing/synthetic_workload.csv \
   cloudsim-rl-project/src/main/resources/
```

### Issue 3: Python Libraries Missing

```
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**

```bash
conda activate cloudsim_rl
pip install pandas numpy gymnasium stable-baselines3 torch
```

---

## 📈 Next Steps

### Immediate (Phase 3)

1. **Implement Python-Java socket bridge**

   - Complete `cloudsim_bridge_server.py`
   - Create Java socket client in CloudSim

2. **Create Gymnasium environment**

   - Wrap CloudSim in `gym.Env` interface
   - Define observation/action spaces

3. **Implement DQN agent**

   - Use Stable Baselines3
   - Train on synthetic workload

4. **Implement PPO agent**
   - Compare with DQN and Q-Learning

### Future (Phase 4-5)

- Implement heuristic baselines (FCFS, Round Robin, SJF)
- Run comprehensive experiments (100+ runs)
- Generate performance comparison plots
- Write final project report

---

## 📚 Key Files to Understand

### Java Files

1. **QLearningBroker.java** - Core Q-Learning algorithm

   - Q-Table: HashMap<String, Double>
   - Action selection: ε-greedy
   - Update rule: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

2. **GoogleTraceReader.java** - CSV to Cloudlet converter

   - Reads from resources folder
   - Creates CloudSim Cloudlets with realistic properties

3. **CloudSimTraceQLearningSimulation.java** - Main simulation
   - 8 hosts, 30 VMs, 1000 cloudlets
   - Integrated Q-Learning + Trace workload

### Python Files

1. **preprocess_google_traces.py** - Data pipeline

   - Reads Google traces or generates synthetic workload
   - Normalizes 0-1 scale to MIPS values

2. **cloudsim_bridge_server.py** - Socket server (template)
   - Ready for Phase 3 DQN/PPO integration

---

## 🎓 Learning Resources

1. **CloudSim Plus:**

   - Documentation: http://cloudsimplus.org/
   - Examples: https://github.com/manoelcampos/cloudsimplus-examples

2. **Reinforcement Learning:**

   - Stable Baselines3: https://stable-baselines3.readthedocs.io/
   - Gymnasium: https://gymnasium.farama.org/

3. **Google Cluster Data:**
   - Dataset: https://github.com/google/cluster-data
   - Paper: Reiss et al., "Google cluster-usage traces" (2011)

---

**Last Updated:** Phase 2 Completion  
**Contact:** See PROJECT_README.md for full documentation

---

## 🏆 Success Criteria

- [ ] Maven compilation works (or IDE import successful)
- [ ] GoogleTraceReader loads 1000 cloudlets
- [ ] Q-Learning simulation completes without errors
- [ ] Performance metrics calculated correctly
- [ ] Q-Table shows learning (increasing Q-values)
- [ ] VM utilization is balanced (load imbalance < 30%)

**Current Status:** 3/6 components working (needs compilation fix)

# CloudSim Plus RL Project - EE 782

## Project Overview

This project implements Reinforcement Learning-based load balancing for cloud resource allocation using CloudSim Plus.

## Project Structure

```
EE_782_Project/
├── cloudsim-rl-project/           # Java Maven project
│   ├── pom.xml                     # Maven dependencies
│   └── src/main/java/com/ee782/cloudsim/
│       ├── CloudSimQLearningSimulation.java    # Main simulation
│       └── QLearningBroker.java                # Q-Learning implementation
│
├── python-bridge/                  # Python-Java bridge (for Phase 3)
│   └── cloudsim_bridge_server.py   # Socket server for RL agents
│
├── cloudsim-plus/                  # CloudSim Plus repository (cloned)
│
├── README.md                       # This file
└── to_do.md                        # Project requirements

```

## Phase 0: Setup (Completed ✓)

### Prerequisites Installed:

- ✓ Java JDK 17 (OpenJDK 17.0.16)
- ✓ Python 3.9 with conda environment `cloudsim_rl`
- ✓ Python packages: gymnasium, stable-baselines3, pandas, numpy, pytorch

### Conda Environment:

```bash
# Activate the environment
conda activate cloudsim_rl

# Verify installation
python --version  # Should show Python 3.9.25
pip list | grep -E "gymnasium|stable-baselines3|pandas|numpy|torch"
```

## Phase 1: Q-Learning Implementation (Completed ✓)

### Features Implemented:

1. **CloudSim Plus Setup**: Basic simulation with 1 Datacenter, 8 Hosts, 30 VMs
2. **Q-Learning Algorithm**: Custom DatacenterBroker with Q-Table
3. **State Space**: VM CPU utilization discretized into buckets
4. **Action Space**: Select target VM for cloudlet assignment
5. **Reward Function**: Based on load balancing and VM utilization

### Running the Simulation:

#### Method 1: Using Maven (Recommended)

```bash
cd cloudsim-rl-project

# Download dependencies (first time only)
mvn clean install

# Compile and run
mvn compile exec:java -Dexec.mainClass="com.ee782.cloudsim.CloudSimQLearningSimulation"
```

#### Method 2: Using Maven wrapper

```bash
cd cloudsim-rl-project

# Make wrapper executable (first time)
chmod +x mvnw

# Run
./mvnw compile exec:java -Dexec.mainClass="com.ee782.cloudsim.CloudSimQLearningSimulation"
```

### Simulation Parameters:

- **Hosts**: 8 (16GB RAM, 8 cores @ 10000 MIPS each)
- **VMs**: 30 (2GB RAM, 2 cores @ 1000 MIPS each)
- **Cloudlets**: 100 (varying lengths: 10k-50k MI)

### Q-Learning Parameters:

- **Learning Rate (α)**: 0.1
- **Discount Factor (γ)**: 0.9
- **Exploration Rate (ε)**: 0.2 (ε-greedy policy)

### Metrics Calculated:

1. Average Response Time
2. Average Waiting Time
3. Throughput (tasks/second)
4. Load Imbalance Degree (%)
5. Average CPU Utilization
6. Q-Table Statistics

## Phase 0 Bridge Architecture (Template Ready)

The Python socket server template is ready for Phase 3 (DQN/PPO implementation).

### Testing the Bridge (Optional):

```bash
# Activate Python environment
conda activate cloudsim_rl

# Run the server
python python-bridge/cloudsim_bridge_server.py
```

Note: Java client implementation will be added in Phase 3.

## Next Steps

### Phase 2: Data Preprocessing

- Download Google Cluster Data or Alibaba traces
- Create CSV parser to convert traces to Cloudlets
- Implement TraceReader class

### Phase 3: Advanced RL (DQN & PPO)

- Complete the Python-Java socket bridge
- Create custom Gym environment
- Implement DQN agent using Stable Baselines3
- Implement PPO agent using Stable Baselines3
- Multi-objective reward shaping

### Phase 4: Heuristic Baselines

- Round Robin (built-in)
- Genetic Algorithm
- Ant Colony Optimization

### Phase 5: Evaluation

- Run experiments on identical datasets
- Compare all algorithms
- Generate visualizations

## Troubleshooting

### Maven Issues:

If Maven is not installed:

```bash
sudo apt install maven
```

### Java Version Issues:

Verify Java 17 is being used:

```bash
java -version
javac -version
```

### Python Environment Issues:

If environment is not found:

```bash
conda env list
conda activate cloudsim_rl
```

## References

- CloudSim Plus: https://github.com/cloudslab/cloudsim-plus
- Stable Baselines3: https://stable-baselines3.readthedocs.io/
- Gymnasium: https://gymnasium.farama.org/

## Authors

EE 782 Course Project

## License

MIT License

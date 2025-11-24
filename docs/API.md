# API Documentation

## Python API

### CloudSimEnv

Custom Gymnasium environment for interacting with CloudSim Plus.

```python
from utils.cloudsim_env import CloudSimEnv

# Initialize environment
env = CloudSimEnv(num_vms=20, host='localhost', port=9999)

# Reset environment
state, info = env.reset()

# Take action
next_state, reward, terminated, truncated, info = env.step(action)

# Close environment
env.close()
```

#### Methods

##### `__init__(num_vms=20, host='localhost', port=9999)`
Initialize the environment.

**Parameters**:
- `num_vms` (int): Number of VMs in the datacenter
- `host` (str): CloudSim server hostname
- `port` (int): CloudSim server port

##### `reset(seed=None, options=None)`
Reset the environment to initial state.

**Returns**:
- `state` (np.ndarray): Initial state observation
- `info` (dict): Additional information

##### `step(action)`
Execute one timestep.

**Parameters**:
- `action` (int): VM index to assign the next cloudlet

**Returns**:
- `state` (np.ndarray): Next state observation
- `reward` (float): Reward for the action
- `terminated` (bool): Whether episode ended
- `truncated` (bool): Whether episode was truncated
- `info` (dict): Additional information

##### `close()`
Close the environment and disconnect from server.

---

### DQNAgent

Deep Q-Network agent for task scheduling.

```python
from algorithms.rl.dqn.dqn_agent import DQNAgent

# Initialize agent
agent = DQNAgent(
    state_dim=21,
    action_dim=20,
    learning_rate=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    buffer_size=10000,
    batch_size=64,
    target_update_freq=10
)

# Select action
action = agent.select_action(state, training=True)

# Store experience
agent.store_experience(state, action, reward, next_state, done)

# Train
loss = agent.train_step()

# Save/Load model
agent.save('model.pth')
agent.load('model.pth')
```

#### Methods

##### `__init__(...)`
Initialize the DQN agent.

**Parameters**:
- `state_dim` (int): Dimension of state space
- `action_dim` (int): Number of possible actions
- `learning_rate` (float): Learning rate for optimizer
- `gamma` (float): Discount factor
- `epsilon_start` (float): Initial exploration rate
- `epsilon_end` (float): Minimum exploration rate
- `epsilon_decay` (float): Decay rate for epsilon
- `buffer_size` (int): Replay buffer capacity
- `batch_size` (int): Training batch size
- `target_update_freq` (int): Target network update frequency

##### `select_action(state, training=True)`
Select action using epsilon-greedy policy.

**Parameters**:
- `state` (np.ndarray): Current state
- `training` (bool): Whether in training mode

**Returns**:
- `action` (int): Selected action

##### `store_experience(state, action, reward, next_state, done)`
Store transition in replay buffer.

##### `train_step()`
Perform one training step.

**Returns**:
- `loss` (float): Training loss, or None if buffer too small

##### `save(path)`
Save model checkpoint.

##### `load(path)`
Load model checkpoint.

---

## Java API

### CloudSimSocketServer

Main simulation server that communicates with Python agents.

```java
public class CloudSimSocketServer {
    private static final int PORT = 9999;
    private static final int NUM_HOSTS = 5;
    private static final int NUM_VMS = 20;
    private static final int CLOUDLETS_PER_EPISODE = 100;
    
    public static void main(String[] args) {
        CloudSimSocketServer server = new CloudSimSocketServer();
        server.startServer();
    }
}
```

#### Protocol

##### Reset Command

**Request**:
```json
{
  "command": "reset"
}
```

**Response**:
```json
{
  "state": [0.0, 0.0, ..., 0.5],
  "info": "Environment reset"
}
```

##### Step Command

**Request**:
```json
{
  "command": "step",
  "action": 5
}
```

**Response**:
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

##### Close Command

**Request**:
```json
{
  "command": "close"
}
```

**Response**:
```json
{}
```

#### State Representation

State is a vector of length `num_vms + 1`:
- First `num_vms` elements: Normalized load of each VM
- Last element: Normalized length of next cloudlet

```
state[i] = vm_load[i] / vm_mips[i]  for i in [0, num_vms)
state[num_vms] = next_cloudlet_length / 100000.0
```

#### Reward Calculation

```java
private double calculateReward(Cloudlet cloudlet, Vm vm) {
    // Response time component
    double estimatedResponseTime = cloudlet.getLength() / vm.getMips();
    double timeReward = -estimatedResponseTime / 100.0;
    
    // Load balancing component
    double[] loads = getVmLoads();
    double loadVariance = calculateVariance(loads);
    double balanceReward = -loadVariance / 1000.0;
    
    // Combined reward
    return 0.7 * timeReward + 0.3 * balanceReward;
}
```

---

## Training Scripts

### train_dqn.py

Train a DQN agent.

```bash
python train_dqn.py [OPTIONS]
```

**Options**:
- `--episodes` (int): Number of training episodes [default: 500]
- `--save-freq` (int): Model save frequency [default: 50]
- `--log-freq` (int): Logging frequency [default: 10]
- `--model-path` (str): Custom model save path [optional]

**Example**:
```bash
python train_dqn.py --episodes 1000 --save-freq 100 --log-freq 20
```

**Output Files**:
- `results/models/dqn_cloudsim_TIMESTAMP.pth`: Model checkpoint
- `results/models/dqn_cloudsim_TIMESTAMP_log.json`: Training log

---

### test_dqn.py

Evaluate a trained DQN agent.

```bash
python test_dqn.py --model-path MODEL_PATH [OPTIONS]
```

**Options**:
- `--model-path` (str): Path to trained model [required]
- `--episodes` (int): Number of evaluation episodes [default: 10]

**Example**:
```bash
python test_dqn.py --model-path ../../../results/models/dqn_cloudsim_20251124.pth --episodes 10
```

**Output Files**:
- `results/models/dqn_cloudsim_TIMESTAMP_eval_results.json`: Evaluation results

---

## Visualization

### visualization.py

Generate plots from training/evaluation logs.

```bash
python visualization.py [OPTIONS]
```

**Options**:
- `--log-path` (str): Path to training log JSON
- `--eval-path` (str): Path to evaluation results JSON
- `--save-dir` (str): Directory to save plots [default: ../results/plots]

**Example**:
```bash
python visualization.py \
    --log-path ../results/models/dqn_cloudsim_20251124_log.json \
    --save-dir ../results/plots
```

**Output Files**:
- `results/plots/training_results.png`: Training curves
- `results/plots/evaluation_results.png`: Performance comparison

---

## Configuration

### Environment Variables

```bash
# Java environment
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Python environment
export PYTHONPATH=/home/anupam/win_desktop/EE_782_Project:$PYTHONPATH

# CloudSim server
export CLOUDSIM_HOST=localhost
export CLOUDSIM_PORT=9999
```

### Python Dependencies

```
numpy>=1.24.0        # Numerical computations
torch>=2.0.0         # Deep learning framework
gymnasium>=0.29.0    # RL environment interface
matplotlib>=3.7.0    # Plotting
seaborn>=0.12.0      # Statistical visualization
pandas>=2.0.0        # Data manipulation
```

### Java Dependencies

```xml
<dependency>
    <groupId>org.cloudsimplus</groupId>
    <artifactId>cloudsim-plus</artifactId>
    <version>8.0.0</version>
</dependency>
<dependency>
    <groupId>com.google.code.gson</groupId>
    <artifactId>gson</artifactId>
    <version>2.10.1</version>
</dependency>
```

---

## Error Handling

### Common Errors

#### Connection Refused

```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Solution**: Ensure CloudSim server is running before starting training.

```bash
# Terminal 1
cd simulation/java
mvn exec:java -Dexec.mainClass="org.ee782.CloudSimSocketServer"
```

#### JAVA_HOME Not Set

```
The JAVA_HOME environment variable is not defined correctly
```

**Solution**: Set JAVA_HOME environment variable.

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
```

#### Import Error

```
ImportError: No module named 'torch'
```

**Solution**: Install Python dependencies.

```bash
conda activate ee782
pip install -r requirements.txt
```

---

## Performance Tuning

### DQN Hyperparameter Tuning

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| learning_rate | 1e-3 | [1e-4, 1e-2] | Convergence speed |
| gamma | 0.99 | [0.9, 0.999] | Long-term planning |
| epsilon_decay | 0.995 | [0.99, 0.999] | Exploration-exploitation |
| batch_size | 64 | [32, 256] | Training stability |
| buffer_size | 10000 | [5000, 50000] | Sample diversity |

### CloudSim Configuration

Edit `CloudSimSocketServer.java`:

```java
private static final int NUM_HOSTS = 5;      // Datacenter hosts
private static final int NUM_VMS = 20;       // Virtual machines
private static final int CLOUDLETS_PER_EPISODE = 100;  // Tasks per episode
```

### System Resources

Recommended:
- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: 4+ cores
- **GPU**: CUDA-compatible GPU for faster training (optional)

---

## Logging

### Training Logs

```json
{
  "training_log": [
    {
      "episode": 10,
      "avg_reward": -2.345,
      "avg_length": 98.2,
      "avg_loss": 0.0234,
      "epsilon": 0.900
    }
  ],
  "episode_rewards": [-2.1, -2.3, ...],
  "episode_lengths": [100, 98, ...],
  "episode_losses": [0.023, 0.024, ...],
  "total_episodes": 500,
  "total_time": 3456.78
}
```

### Evaluation Results

```json
{
  "episode_rewards": [-1.2, -1.5, ...],
  "episode_lengths": [100, 100, ...],
  "episode_infos": [
    {
      "avg_response_time": 12.3,
      "sla_violations": 3,
      "sla_violation_rate": 0.032
    }
  ],
  "summary": {
    "mean_reward": -1.35,
    "std_reward": 0.15,
    "mean_length": 100.0,
    "std_length": 0.0
  }
}
```

---

## Testing

### Unit Tests (Future)

```bash
# Run Python tests
pytest tests/

# Run Java tests
cd simulation/java
mvn test
```

### Integration Tests

```bash
# Test CloudSim server
cd simulation/java
mvn exec:java -Dexec.mainClass="org.ee782.CloudSimSocketServer"

# Test Python client (in another terminal)
python -c "
from utils.cloudsim_env import CloudSimEnv
env = CloudSimEnv()
state, _ = env.reset()
print('State:', state)
env.close()
"
```

---

## Troubleshooting

See [SETUP_GUIDE.md](../SETUP_GUIDE.md) for detailed troubleshooting steps.

Quick checks:
1. Verify Java version: `java -version` (should be 17+)
2. Verify Maven: `mvn -version`
3. Verify Python: `python --version` (should be 3.9+)
4. Check CloudSim build: `cd simulation/java && mvn compile`
5. Test Python imports: `python -c "import torch; import gymnasium"`

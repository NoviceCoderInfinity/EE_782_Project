# Algorithm Details & Implementation Notes

## Table of Contents
1. [DQN Implementation](#dqn-implementation)
2. [PPO Implementation](#ppo-implementation)
3. [Q-Learning Implementation](#q-learning-implementation)
4. [Heuristic Algorithms](#heuristic-algorithms)

---

## DQN Implementation

### Network Architecture

```python
class DQNNetwork(nn.Module):
    Input Layer:  state_dim (21)
    Hidden Layer 1: 128 neurons + ReLU
    Hidden Layer 2: 128 neurons + ReLU
    Output Layer: action_dim (20)
```

### Training Algorithm

```
for episode in range(num_episodes):
    state = env.reset()
    
    for step in range(max_steps):
        # Select action using epsilon-greedy
        if random() < epsilon:
            action = random_action()
        else:
            action = argmax(Q(state))
        
        # Execute action
        next_state, reward, done = env.step(action)
        
        # Store transition
        replay_buffer.push(state, action, reward, next_state, done)
        
        # Train if enough samples
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            
            # Compute Q-values
            Q_current = Q_network(states)[actions]
            Q_target = rewards + gamma * max(Target_network(next_states))
            
            # Update network
            loss = MSE(Q_current, Q_target)
            optimizer.step()
        
        # Update target network periodically
        if step % target_update_freq == 0:
            Target_network = Q_network
        
        # Decay epsilon
        epsilon *= epsilon_decay
        
        state = next_state
```

### Reward Design

Multi-objective reward balancing:
- **Response Time Component** (70%): Minimizes task execution time
- **Load Balancing Component** (30%): Ensures even VM utilization

```python
response_time_reward = -estimated_response_time / 100
load_balance_reward = -variance(vm_loads) / 1000
total_reward = 0.7 * response_time_reward + 0.3 * load_balance_reward
```

### Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 1e-3 | Standard for Adam optimizer |
| Gamma (γ) | 0.99 | High discount for long-term planning |
| Epsilon Start | 1.0 | Full exploration initially |
| Epsilon End | 0.01 | Maintain minimal exploration |
| Epsilon Decay | 0.995 | Gradual transition |
| Batch Size | 64 | Balance between speed and stability |
| Replay Buffer | 10,000 | Sufficient for task diversity |
| Target Update | 10 steps | Frequent enough for stability |

### Experience Replay

Benefits:
- Breaks correlation between consecutive samples
- Enables learning from rare events multiple times
- Improves sample efficiency

Implementation:
```python
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

---

## PPO Implementation (Planned)

### Actor-Critic Architecture

```python
class PPONetwork:
    Actor Network:
        Input → FC(128) → ReLU → FC(128) → ReLU → Softmax(20)
        Outputs: Action probabilities
    
    Critic Network:
        Input → FC(128) → ReLU → FC(128) → ReLU → FC(1)
        Outputs: State value V(s)
```

### PPO Objective Function

```
L(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]

where:
- r(θ) = π_θ(a|s) / π_θ_old(a|s)  (probability ratio)
- A = advantage estimate
- ε = clipping parameter (0.2)
```

### Advantages over DQN

- More stable training
- Better for continuous action spaces
- No need for experience replay
- Direct policy optimization

---

## Q-Learning Implementation

### Q-Table Structure

```
State Space: Discretized VM loads (LOW/MED/HIGH per VM + cloudlet size)
Action Space: 20 VMs
Q-Table: Dictionary (defaultdict) mapping (state, action) → Q-value
Storage: Pickle format for persistence
```

### Update Rule

```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

Parameters:
- α (learning rate) = 0.1
- γ (discount) = 0.9
- ε_start (epsilon) = 1.0
- ε_end = 0.01
- ε_decay = 0.995
```

### Training Algorithm

```
for episode in range(num_episodes):
    state = env.reset()
    
    for step in range(max_steps):
        # Discretize continuous state
        discrete_state = discretize_state(state)
        
        # Select action using epsilon-greedy
        if random() < epsilon:
            action = random_action()
        else:
            action = argmax(Q_table[discrete_state])
        
        # Execute action
        next_state, reward, done = env.step(action)
        discrete_next_state = discretize_state(next_state)
        
        # Q-Learning update
        current_q = Q_table[discrete_state][action]
        max_next_q = max(Q_table[discrete_next_state])
        new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
        Q_table[discrete_state][action] = new_q
        
        # Decay epsilon
        epsilon *= epsilon_decay
        
        state = next_state
```

### State Discretization

```python
def discretize_state(state):
    discrete_state = []
    
    # Discretize VM loads (first 20 elements)
    for i in range(num_vms):
        load = state[i]
        if load < 0.33:
            discrete_state.append('LOW')
        elif load < 0.67:
            discrete_state.append('MED')
        else:
            discrete_state.append('HIGH')
    
    # Discretize cloudlet length (last element)
    cloudlet_length = state[-1]
    if cloudlet_length < 0.33:
        discrete_state.append('SMALL')
    elif cloudlet_length < 0.67:
        discrete_state.append('MEDIUM')
    else:
        discrete_state.append('LARGE')
    
    return tuple(discrete_state)
```

### Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate (α) | 0.1 | Standard for tabular Q-learning |
| Gamma (γ) | 0.9 | Moderate discounting |
| Epsilon Start | 1.0 | Full exploration initially |
| Epsilon End | 0.01 | Minimal exploration |
| Epsilon Decay | 0.995 | Gradual transition |
| Discretization Bins | 3 | Balance expressiveness vs. table size |

### Advantages

- **Simple**: No neural networks, direct Q-value storage
- **Interpretable**: Can inspect Q-values directly
- **Fast training**: No backpropagation overhead
- **Guaranteed convergence**: Under certain conditions

### Limitations

- **Curse of dimensionality**: State space grows exponentially
- **Discretization loss**: May lose important state distinctions
- **Memory usage**: Large state spaces need large tables
- **Generalization**: No generalization to unseen states

---

## Heuristic Algorithms

### 1. Round Robin

**Algorithm**:
```python
current_vm = 0

def schedule(cloudlet):
    selected_vm = vm_list[current_vm]
    current_vm = (current_vm + 1) % num_vms
    return selected_vm
```

**Pros**:
- Simple
- Fair distribution
- Low overhead

**Cons**:
- Ignores VM capabilities
- No load consideration
- Suboptimal performance

### 2. Genetic Algorithm

**Chromosome Representation**:
```
Chromosome = [VM_id for each cloudlet]
Example: [3, 7, 2, 15, 8, ...]  (100 genes for 100 cloudlets)
```

**Fitness Function**:
```python
def fitness(chromosome):
    makespan = calculate_makespan(chromosome)
    load_balance = calculate_load_balance(chromosome)
    return 1 / (makespan + load_balance_penalty)
```

**Genetic Operators**:
```python
# Crossover (Single Point)
parent1 = [3, 7, 2, | 15, 8, 12]
parent2 = [5, 1, 9, | 3, 7, 2]
child   = [3, 7, 2, | 3, 7, 2]

# Mutation (Random Reset)
before = [3, 7, 2, 15, 8]
after  = [3, 7, 19, 15, 8]  # Gene 2 mutated
```

**Parameters**:
- Population Size: 100
- Generations: 50
- Crossover Rate: 0.8
- Mutation Rate: 0.1
- Selection: Tournament (size=3)

### 3. Ant Colony Optimization

**Pheromone Model**:
```
τ[vm_id][cloudlet_id] = pheromone strength
Higher τ → More likely to assign cloudlet to that VM
```

**Probability of Selection**:
```
P(vm_j for cloudlet_i) = [τ(i,j)^α × η(i,j)^β] / Σ[τ(i,k)^α × η(i,k)^β]

where:
- τ(i,j) = pheromone level
- η(i,j) = heuristic information (1/expected_time)
- α = pheromone importance (1.0)
- β = heuristic importance (2.0)
```

**Pheromone Update**:
```
τ(i,j) ← (1-ρ)τ(i,j) + Δτ(i,j)

where:
- ρ = evaporation rate (0.1)
- Δτ(i,j) = Q / cost (if ant used this edge)
- Q = constant (100)
```

**Parameters**:
- Number of Ants: 20
- Iterations: 50
- α (pheromone weight): 1.0
- β (heuristic weight): 2.0
- ρ (evaporation): 0.1

---

## Comparative Analysis

### Time Complexity

| Algorithm | Training Time | Inference Time | Space Complexity |
|-----------|---------------|----------------|------------------|
| DQN | O(episodes × steps × batch) | O(1) | O(buffer_size) |
| PPO | O(episodes × steps) | O(1) | O(trajectory_length) |
| Q-Learning | O(episodes × steps) | O(1) | O(state_space) |
| Round Robin | N/A | O(1) | O(1) |
| GA | O(generations × population × eval) | O(1) | O(population) |
| ACO | O(iterations × ants × cloudlets) | O(1) | O(cloudlets × VMs) |

### Expected Performance (Hypothetical)

Based on literature and typical results:

| Algorithm | Avg Response Time | SLA Violations | Adaptability |
|-----------|-------------------|----------------|--------------|
| DQN | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| PPO | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐⭐ |
| Q-Learning | ⭐⭐⭐☆☆ | ⭐⭐⭐☆☆ | ⭐⭐⭐☆☆ |
| Round Robin | ⭐⭐☆☆☆ | ⭐⭐☆☆☆ | ⭐☆☆☆☆ |
| GA | ⭐⭐⭐⭐☆ | ⭐⭐⭐☆☆ | ⭐⭐☆☆☆ |
| ACO | ⭐⭐⭐⭐☆ | ⭐⭐⭐☆☆ | ⭐⭐☆☆☆ |

---

## Implementation Roadmap

### Phase 1 ✅: DQN
- [x] Network architecture
- [x] Experience replay
- [x] Target network
- [x] Training loop
- [x] Evaluation metrics

### Phase 2 🔨: PPO
- [ ] Actor-Critic networks
- [ ] Advantage estimation (GAE)
- [ ] Clipped objective
- [ ] Training loop

### Phase 3 🔨: Baselines
- [ ] Q-Learning with discretization
- [ ] Round Robin
- [ ] GA implementation
- [ ] ACO implementation

### Phase 4 🔨: Optimization
- [ ] Hyperparameter tuning
- [ ] Multi-objective optimization
- [ ] Transfer learning
- [ ] Ensemble methods

---

## References

1. **DQN**: Mnih et al., "Human-level control through deep reinforcement learning", Nature, 2015
2. **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms", arXiv, 2017
3. **GA**: Holland, "Adaptation in Natural and Artificial Systems", 1975
4. **ACO**: Dorigo et al., "Ant Colony Optimization", MIT Press, 2004

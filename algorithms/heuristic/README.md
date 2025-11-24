# Heuristic Scheduling Algorithms

Traditional and bio-inspired algorithms for cloud task scheduling.

## Algorithms

### 1. Round-Robin
Simple cyclic task distribution ensuring fair allocation.

**Usage:**
```python
from algorithms.heuristic.round_robin import RoundRobinScheduler

scheduler = RoundRobinScheduler(num_vms=20)
action = scheduler.select_action()
```

### 2. Least Connection
Assigns tasks to VM with lowest current load.

**Usage:**
```python
from algorithms.heuristic.least_connection import LeastConnectionScheduler

scheduler = LeastConnectionScheduler(num_vms=20)
action = scheduler.select_action(state)
```

### 3. Weighted Round-Robin
Distributes tasks proportionally based on VM capacities.

**Usage:**
```python
from algorithms.heuristic.weighted_round_robin import WeightedRoundRobinScheduler

scheduler = WeightedRoundRobinScheduler(num_vms=20)
action = scheduler.select_action()
```

### 4. Genetic Algorithm (GA)
Evolves scheduling solutions using selection, crossover, and mutation.

**Parameters:**
- Population size: 50
- Generations: 100
- Crossover rate: 0.8
- Mutation rate: 0.1

**Usage:**
```python
from algorithms.heuristic.genetic_algorithm import GeneticAlgorithmScheduler

scheduler = GeneticAlgorithmScheduler(num_vms=20, population_size=50, generations=100)
action = scheduler.select_action(state, task_queue)
```

### 5. Ant Colony Optimization (ACO)
Uses pheromone trails to discover optimal scheduling paths.

**Parameters:**
- Number of ants: 20
- Iterations: 50
- Evaporation rate: 0.5
- Alpha (pheromone importance): 1.0
- Beta (heuristic importance): 2.0

**Usage:**
```python
from algorithms.heuristic.ant_colony_optimization import AntColonyOptimizationScheduler

scheduler = AntColonyOptimizationScheduler(num_vms=20, num_ants=20, iterations=50)
action = scheduler.select_action(state, task_queue)
```

## Performance Comparison

Based on experimental results:

| Algorithm | Avg Response Time | CPU Utilization | SLA Compliance |
|-----------|-------------------|-----------------|----------------|
| Round-Robin | 15.42s | 58.3% | 82.5% |
| Least Conn. | 14.22s | 62.7% | 85.3% |
| Weighted RR | 13.74s | -- | -- |
| GA | 12.78s | 66.8% | 88.9% |
| ACO | 12.53s | -- | -- |

## Notes

- All schedulers implement `select_action(state)` interface
- GA and ACO benefit from task queue lookahead
- For production use, consider tuning hyperparameters based on workload characteristics

"""
Ant Colony Optimization for Cloud Task Scheduling

Uses artificial ant behavior and pheromone trails to discover
optimal task-to-VM assignment paths.
"""

import numpy as np
import random


class AntColonyOptimizationScheduler:
    """
    ACO scheduler for cloud task scheduling
    
    Simulates ant colony behavior to find optimal scheduling solutions.
    """
    
    def __init__(
        self,
        num_vms=20,
        num_ants=20,
        iterations=50,
        alpha=1.0,
        beta=2.0,
        evaporation_rate=0.5,
        pheromone_deposit=1.0
    ):
        """
        Initialize ACO scheduler
        
        Args:
            num_vms: Number of virtual machines
            num_ants: Number of ants in colony
            iterations: Number of optimization iterations
            alpha: Pheromone importance factor
            beta: Heuristic importance factor
            evaporation_rate: Pheromone evaporation rate
            pheromone_deposit: Amount of pheromone deposited
        """
        self.num_vms = num_vms
        self.num_ants = num_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        
        # Pheromone matrix: [task_position][vm_id]
        self.pheromone = None
        
        # Best solution found
        self.best_solution = None
        self.best_cost = float('inf')
        
        # Statistics
        self.iteration_count = 0
    
    def _initialize_pheromone(self, num_tasks):
        """
        Initialize pheromone trails
        
        Args:
            num_tasks: Number of tasks to schedule
        """
        # Initialize with small uniform pheromone
        self.pheromone = np.ones((num_tasks, self.num_vms)) * 0.1
    
    def _calculate_heuristic(self, vm_loads):
        """
        Calculate heuristic desirability (inverse of load)
        
        Args:
            vm_loads: Current VM loads
        
        Returns:
            Heuristic values for each VM
        """
        # Prefer VMs with lower load
        # Add small epsilon to avoid division by zero
        heuristic = 1.0 / (vm_loads + 0.01)
        return heuristic
    
    def _select_vm(self, task_idx, vm_loads):
        """
        Select VM based on pheromone and heuristic
        
        Args:
            task_idx: Current task index
            vm_loads: Current VM loads
        
        Returns:
            Selected VM index
        """
        # Get pheromone and heuristic values
        pheromone_values = self.pheromone[task_idx] ** self.alpha
        heuristic_values = self._calculate_heuristic(vm_loads) ** self.beta
        
        # Calculate probabilities
        probabilities = pheromone_values * heuristic_values
        probabilities = probabilities / np.sum(probabilities)
        
        # Select VM based on probabilities
        vm_id = np.random.choice(self.num_vms, p=probabilities)
        return vm_id
    
    def _calculate_cost(self, solution, vm_loads, task_lengths):
        """
        Calculate cost of a solution
        
        Args:
            solution: Task-to-VM assignment
            vm_loads: Initial VM loads
            task_lengths: Task length array
        
        Returns:
            Solution cost (lower is better)
        """
        # Calculate resulting VM loads
        vm_total_load = vm_loads.copy()
        for task_idx, vm_id in enumerate(solution):
            if task_idx < len(task_lengths):
                vm_total_load[vm_id] += task_lengths[task_idx]
        
        # Cost components
        makespan = np.max(vm_total_load)  # Maximum completion time
        load_variance = np.var(vm_total_load)  # Load imbalance
        avg_load = np.mean(vm_total_load)  # Average load
        
        # Combined cost
        cost = 0.5 * makespan + 0.3 * load_variance + 0.2 * avg_load
        return cost
    
    def _update_pheromone(self, solutions, costs):
        """
        Update pheromone trails based on ant solutions
        
        Args:
            solutions: List of ant solutions
            costs: List of solution costs
        """
        # Evaporation
        self.pheromone *= (1 - self.evaporation_rate)
        
        # Deposit pheromone
        for solution, cost in zip(solutions, costs):
            # Better solutions deposit more pheromone
            deposit_amount = self.pheromone_deposit / (cost + 1e-6)
            
            for task_idx, vm_id in enumerate(solution):
                self.pheromone[task_idx][vm_id] += deposit_amount
        
        # Ensure minimum pheromone level
        self.pheromone = np.maximum(self.pheromone, 0.01)
    
    def optimize(self, vm_loads, task_queue):
        """
        Run ACO optimization
        
        Args:
            vm_loads: Current VM loads
            task_queue: Queue of tasks to schedule
        
        Returns:
            Best VM assignment for first task
        """
        num_tasks = min(len(task_queue), 10)  # Schedule next 10 tasks
        if num_tasks == 0:
            return random.randint(0, self.num_vms - 1)
        
        task_lengths = np.array([task_queue[i] for i in range(num_tasks)])
        
        # Initialize pheromone
        self._initialize_pheromone(num_tasks)
        
        # ACO iterations
        for iteration in range(self.iterations):
            solutions = []
            costs = []
            
            # Each ant constructs a solution
            for ant in range(self.num_ants):
                solution = []
                temp_vm_loads = vm_loads.copy()
                
                # Construct solution task by task
                for task_idx in range(num_tasks):
                    vm_id = self._select_vm(task_idx, temp_vm_loads)
                    solution.append(vm_id)
                    
                    # Update temporary loads
                    if task_idx < len(task_lengths):
                        temp_vm_loads[vm_id] += task_lengths[task_idx]
                
                # Evaluate solution
                cost = self._calculate_cost(solution, vm_loads, task_lengths)
                solutions.append(solution)
                costs.append(cost)
                
                # Update best solution
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = solution.copy()
            
            # Update pheromone trails
            self._update_pheromone(solutions, costs)
            self.iteration_count += 1
        
        # Return first assignment from best solution
        return int(self.best_solution[0]) if self.best_solution is not None else 0
    
    def select_action(self, state, task_queue=None):
        """
        Select VM for next task using ACO
        
        Args:
            state: Current state [vm_loads, cloudlet_length]
            task_queue: Future task lengths (if available)
        
        Returns:
            Selected VM index
        """
        # Extract VM loads from state
        vm_loads = state[:self.num_vms]
        
        # If task queue not provided, use heuristic
        if task_queue is None:
            # Select VM with lowest load (with small randomness)
            heuristic = self._calculate_heuristic(vm_loads)
            if random.random() < 0.2:  # 20% exploration
                return random.randint(0, self.num_vms - 1)
            return int(np.argmax(heuristic))
        
        # Run ACO optimization
        return self.optimize(vm_loads, task_queue)
    
    def reset(self):
        """Reset scheduler state"""
        self.pheromone = None
        self.best_solution = None
        self.best_cost = float('inf')
        self.iteration_count = 0
    
    def get_statistics(self):
        """Get scheduler statistics"""
        return {
            'algorithm': 'Ant-Colony-Optimization',
            'num_vms': self.num_vms,
            'iterations_run': self.iteration_count,
            'best_cost': float(self.best_cost)
        }

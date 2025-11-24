"""
Genetic Algorithm for Cloud Task Scheduling

Evolves scheduling solutions using selection, crossover, and mutation
to find near-optimal task-to-VM assignments.
"""

import numpy as np
import random


class GeneticAlgorithmScheduler:
    """
    Genetic Algorithm scheduler for cloud tasks
    
    Uses evolutionary operators to optimize task assignments.
    """
    
    def __init__(
        self,
        num_vms=20,
        population_size=50,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elite_size=5
    ):
        """
        Initialize GA scheduler
        
        Args:
            num_vms: Number of virtual machines
            population_size: Number of individuals in population
            generations: Number of evolutionary generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elite_size: Number of best individuals to preserve
        """
        self.num_vms = num_vms
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        # Current best solution
        self.best_solution = None
        self.best_fitness = float('-inf')
        
        # Statistics
        self.generation_count = 0
    
    def _initialize_population(self, num_tasks):
        """
        Create initial random population
        
        Args:
            num_tasks: Number of tasks to schedule
        
        Returns:
            Population of random solutions
        """
        population = []
        for _ in range(self.population_size):
            # Random task-to-VM assignment
            individual = np.random.randint(0, self.num_vms, size=num_tasks)
            population.append(individual)
        return population
    
    def _calculate_fitness(self, individual, vm_loads, task_lengths):
        """
        Calculate fitness of a scheduling solution
        
        Args:
            individual: Task-to-VM assignment array
            vm_loads: Current VM load array
            task_lengths: Array of task lengths
        
        Returns:
            Fitness score (higher is better)
        """
        # Calculate resulting VM loads
        vm_total_load = vm_loads.copy()
        for task_idx, vm_id in enumerate(individual):
            if task_idx < len(task_lengths):
                vm_total_load[vm_id] += task_lengths[task_idx]
        
        # Fitness components
        # 1. Minimize maximum load (makespan)
        max_load = np.max(vm_total_load)
        makespan_penalty = -max_load
        
        # 2. Minimize load variance (balance)
        load_variance = np.var(vm_total_load)
        balance_penalty = -load_variance
        
        # 3. Minimize average load
        avg_load = np.mean(vm_total_load)
        avg_penalty = -avg_load
        
        # Combined fitness
        fitness = 0.5 * makespan_penalty + 0.3 * balance_penalty + 0.2 * avg_penalty
        return fitness
    
    def _selection(self, population, fitness_scores):
        """
        Tournament selection
        
        Args:
            population: Current population
            fitness_scores: Fitness of each individual
        
        Returns:
            Selected parent
        """
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1, parent2):
        """
        Single-point crossover
        
        Args:
            parent1: First parent
            parent2: Second parent
        
        Returns:
            Two offspring
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        point = random.randint(1, len(parent1) - 1)
        offspring1 = np.concatenate([parent1[:point], parent2[point:]])
        offspring2 = np.concatenate([parent2[:point], parent1[point:]])
        return offspring1, offspring2
    
    def _mutate(self, individual):
        """
        Random mutation
        
        Args:
            individual: Individual to mutate
        
        Returns:
            Mutated individual
        """
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = random.randint(0, self.num_vms - 1)
        return individual
    
    def evolve(self, vm_loads, task_queue):
        """
        Run GA evolution to find optimal assignment
        
        Args:
            vm_loads: Current VM loads
            task_queue: Queue of tasks to schedule
        
        Returns:
            Best task assignment
        """
        num_tasks = min(len(task_queue), 10)  # Schedule next 10 tasks
        if num_tasks == 0:
            return random.randint(0, self.num_vms - 1)
        
        task_lengths = np.array([task_queue[i] for i in range(num_tasks)])
        
        # Initialize population
        population = self._initialize_population(num_tasks)
        
        # Evolution loop
        for gen in range(self.generations):
            # Calculate fitness
            fitness_scores = [
                self._calculate_fitness(ind, vm_loads, task_lengths)
                for ind in population
            ]
            
            # Track best solution
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_solution = population[best_idx].copy()
            
            # Elitism: preserve best individuals
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            elites = [population[i].copy() for i in elite_indices]
            
            # Create new population
            new_population = elites.copy()
            
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._selection(population, fitness_scores)
                parent2 = self._selection(population, fitness_scores)
                
                # Crossover
                offspring1, offspring2 = self._crossover(parent1, parent2)
                
                # Mutation
                offspring1 = self._mutate(offspring1)
                offspring2 = self._mutate(offspring2)
                
                new_population.extend([offspring1, offspring2])
            
            population = new_population[:self.population_size]
            self.generation_count += 1
        
        # Return first assignment from best solution
        return int(self.best_solution[0]) if self.best_solution is not None else 0
    
    def select_action(self, state, task_queue=None):
        """
        Select VM for next task using GA
        
        Args:
            state: Current state [vm_loads, cloudlet_length]
            task_queue: Future task lengths (if available)
        
        Returns:
            Selected VM index
        """
        # Extract VM loads from state
        vm_loads = state[:self.num_vms]
        
        # If task queue not provided, use simple heuristic
        if task_queue is None:
            # Use least loaded VM with small randomness
            base_action = np.argmin(vm_loads)
            if random.random() < 0.2:  # 20% exploration
                return random.randint(0, self.num_vms - 1)
            return int(base_action)
        
        # Run GA evolution
        return self.evolve(vm_loads, task_queue)
    
    def reset(self):
        """Reset scheduler state"""
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.generation_count = 0
    
    def get_statistics(self):
        """Get scheduler statistics"""
        return {
            'algorithm': 'Genetic-Algorithm',
            'num_vms': self.num_vms,
            'generations_run': self.generation_count,
            'best_fitness': float(self.best_fitness)
        }

"""
Weighted Round-Robin Scheduling Algorithm

Assigns tasks based on VM capacity weights, with higher-capacity VMs
receiving proportionally more tasks.
"""

import numpy as np


class WeightedRoundRobinScheduler:
    """
    Weighted Round-Robin scheduler
    
    Distributes tasks proportionally based on VM weights/capacities.
    """
    
    def __init__(self, num_vms=20, vm_weights=None):
        """
        Initialize Weighted Round-Robin scheduler
        
        Args:
            num_vms: Number of virtual machines
            vm_weights: Array of VM weights (if None, uses uniform weights)
        """
        self.num_vms = num_vms
        
        # Initialize weights (default: uniform)
        if vm_weights is None:
            self.vm_weights = np.ones(num_vms)
        else:
            self.vm_weights = np.array(vm_weights)
        
        # Normalize weights to create probability distribution
        self.probabilities = self.vm_weights / np.sum(self.vm_weights)
        
        # For deterministic weighted RR, create a repeating sequence
        self.sequence = self._create_weighted_sequence()
        self.current_index = 0
    
    def _create_weighted_sequence(self):
        """
        Create weighted sequence for deterministic assignment
        
        Returns:
            Array of VM indices based on weights
        """
        # Convert weights to integer counts (scale by 10)
        counts = (self.vm_weights * 10).astype(int)
        sequence = []
        for vm_id in range(self.num_vms):
            sequence.extend([vm_id] * counts[vm_id])
        return np.array(sequence)
    
    def select_action(self, state=None):
        """
        Select VM based on weighted distribution
        
        Args:
            state: Current state (can be used to adjust weights)
        
        Returns:
            Selected VM index
        """
        # Use deterministic weighted sequence
        if len(self.sequence) > 0:
            action = self.sequence[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.sequence)
        else:
            # Fallback: random selection based on probabilities
            action = np.random.choice(self.num_vms, p=self.probabilities)
        
        return int(action)
    
    def reset(self):
        """Reset scheduler state"""
        self.current_index = 0
    
    def set_weights(self, vm_weights):
        """
        Update VM weights
        
        Args:
            vm_weights: New weight array
        """
        self.vm_weights = np.array(vm_weights)
        self.probabilities = self.vm_weights / np.sum(self.vm_weights)
        self.sequence = self._create_weighted_sequence()
        self.current_index = 0
    
    def get_statistics(self):
        """Get scheduler statistics"""
        return {
            'algorithm': 'Weighted-Round-Robin',
            'num_vms': self.num_vms,
            'weights': self.vm_weights.tolist(),
            'sequence_length': len(self.sequence)
        }

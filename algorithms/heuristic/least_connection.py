"""
Least Connection Scheduling Algorithm

Assigns tasks to the VM with the fewest active connections/tasks,
providing basic load-aware scheduling.
"""

import numpy as np


class LeastConnectionScheduler:
    """
    Least Connection task scheduler
    
    Selects the VM with the lowest current load for task assignment.
    """
    
    def __init__(self, num_vms=20):
        """
        Initialize Least Connection scheduler
        
        Args:
            num_vms: Number of virtual machines
        """
        self.num_vms = num_vms
        self.vm_loads = np.zeros(num_vms)
    
    def select_action(self, state):
        """
        Select VM with lowest load
        
        Args:
            state: Current state [vm_loads, cloudlet_length]
        
        Returns:
            Index of VM with minimum load
        """
        # Extract VM loads from state (first num_vms elements)
        if state is not None and len(state) >= self.num_vms:
            vm_loads = state[:self.num_vms]
        else:
            vm_loads = self.vm_loads
        
        # Select VM with minimum load
        action = np.argmin(vm_loads)
        return int(action)
    
    def reset(self):
        """Reset scheduler state"""
        self.vm_loads = np.zeros(self.num_vms)
    
    def get_statistics(self):
        """Get scheduler statistics"""
        return {
            'algorithm': 'Least-Connection',
            'num_vms': self.num_vms,
            'avg_load': float(np.mean(self.vm_loads)),
            'load_variance': float(np.var(self.vm_loads))
        }

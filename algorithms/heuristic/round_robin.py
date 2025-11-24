"""
Round-Robin Scheduling Algorithm

Simple cyclic task assignment that distributes tasks equally across VMs
without considering their current load or task characteristics.
"""


class RoundRobinScheduler:
    """
    Round-Robin task scheduler
    
    Assigns tasks to VMs in a circular fashion, ensuring fair distribution
    but without load awareness.
    """
    
    def __init__(self, num_vms=20):
        """
        Initialize Round-Robin scheduler
        
        Args:
            num_vms: Number of virtual machines
        """
        self.num_vms = num_vms
        self.current_vm = 0
    
    def select_action(self, state=None):
        """
        Select next VM in round-robin order
        
        Args:
            state: Current state (ignored for round-robin)
        
        Returns:
            Selected VM index
        """
        action = self.current_vm
        self.current_vm = (self.current_vm + 1) % self.num_vms
        return action
    
    def reset(self):
        """Reset scheduler state"""
        self.current_vm = 0
    
    def get_statistics(self):
        """Get scheduler statistics"""
        return {
            'algorithm': 'Round-Robin',
            'num_vms': self.num_vms,
            'current_position': self.current_vm
        }

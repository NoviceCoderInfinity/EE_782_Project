"""
CloudSim Gymnasium Environment
Phase 3: Deep Reinforcement Learning Integration

This module provides a Gymnasium-compatible environment that wraps CloudSim Plus
simulation for training DQN and PPO agents.

Features:
- Socket-based communication with Java CloudSim simulation
- Observation space: VM resource utilization states
- Action space: VM selection for task scheduling
- Reward function: Multi-objective optimization
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import json
import time
from typing import Dict, Tuple, Optional, Any


class CloudSimEnv(gym.Env):
    """
    Custom Gymnasium Environment for CloudSim Plus integration.
    
    Observation Space:
        Box(0, 1, shape=(num_vms * 3,)) - Flattened array of:
        - VM CPU utilization (0-1)
        - VM RAM utilization (0-1)
        - VM bandwidth utilization (0-1)
    
    Action Space:
        Discrete(num_vms) - Select VM ID for task placement
    
    Reward:
        Multi-objective reward combining:
        - Response time (negative)
        - Resource utilization balance
        - SLA violation penalty
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        num_vms: int = 30,
        host: str = 'localhost',
        port: int = 5555,
        max_steps: int = 1000,
        reward_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize CloudSim Gymnasium Environment.
        
        Args:
            num_vms: Number of VMs in the simulation
            host: Java socket server host
            port: Java socket server port
            max_steps: Maximum steps per episode
            reward_weights: Dict with keys 'time', 'balance', 'sla'
        """
        super(CloudSimEnv, self).__init__()
        
        self.num_vms = num_vms
        self.host = host
        self.port = port
        self.max_steps = max_steps
        self.current_step = 0
        
        # Reward function weights
        if reward_weights is None:
            self.reward_weights = {
                'time': 0.5,      # Response time weight
                'balance': 0.3,   # Load balance weight
                'sla': 0.2        # SLA violation weight
            }
        else:
            self.reward_weights = reward_weights
        
        # Define action and observation space
        # Action: Select one of num_vms VMs
        self.action_space = spaces.Discrete(num_vms)
        
        # Observation: CPU, RAM, BW utilization for each VM (normalized 0-1)
        # Shape: (num_vms * 3,) = flattened [cpu1, ram1, bw1, cpu2, ram2, bw2, ...]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_vms * 3,),
            dtype=np.float32
        )
        
        # Socket connection
        self.socket = None
        self.connected = False
        
        # Statistics
        self.episode_count = 0
        self.total_reward = 0.0
        self.cloudlets_completed = 0
        
    def connect(self) -> bool:
        """
        Establish socket connection to Java CloudSim server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.socket:
                self.socket.close()
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(30.0)  # 30 second timeout
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"✓ Connected to CloudSim server at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close socket connection."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
            self.connected = False
    
    def _convert_numpy_types(self, obj):
        """
        Recursively convert numpy types to native Python types for JSON serialization.
        
        Args:
            obj: Object to convert (can be dict, list, numpy type, etc.)
            
        Returns:
            Object with all numpy types converted to Python native types
        """
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _send_message(self, message: Dict) -> Dict:
        """
        Send JSON message to Java server and receive response.
        
        Args:
            message: Dictionary to send as JSON
            
        Returns:
            Response dictionary from server
        """
        if not self.connected:
            if not self.connect():
                raise ConnectionError("Cannot connect to CloudSim server")
        
        try:
            # Convert numpy types to native Python types for JSON serialization
            message = self._convert_numpy_types(message)
            
            # Send message
            message_json = json.dumps(message) + "\n"
            self.socket.sendall(message_json.encode('utf-8'))
            
            # Receive response
            response_data = b""
            while True:
                chunk = self.socket.recv(4096)
                if not chunk:
                    break
                response_data += chunk
                if b"\n" in chunk:  # End of JSON message
                    break
            
            response = json.loads(response_data.decode('utf-8'))
            return response
            
        except Exception as e:
            print(f"✗ Communication error: {e}")
            self.connected = False
            raise
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            observation: Initial state observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_count += 1
        self.total_reward = 0.0
        self.cloudlets_completed = 0
        
        # Send reset command to Java
        message = {
            'command': 'reset',
            'episode': self.episode_count
        }
        
        try:
            response = self._send_message(message)
            
            if response.get('status') != 'success':
                raise RuntimeError(f"Reset failed: {response.get('message')}")
            
            # Extract initial observation
            observation = self._extract_observation(response)
            
            info = {
                'episode': self.episode_count,
                'message': 'Environment reset'
            }
            
            return observation, info
            
        except Exception as e:
            print(f"✗ Reset error: {e}")
            # Return zero observation as fallback
            return np.zeros(self.observation_space.shape, dtype=np.float32), {'error': str(e)}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: VM ID to assign the next cloudlet to (0 to num_vms-1)
            
        Returns:
            observation: New state after action
            reward: Reward for this step
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut off (max steps)
            info: Additional information
        """
        if not isinstance(action, (int, np.integer)):
            action = int(action)
        
        if action < 0 or action >= self.num_vms:
            raise ValueError(f"Invalid action {action}. Must be in range [0, {self.num_vms-1}]")
        
        self.current_step += 1
        
        # Send action to Java
        message = {
            'command': 'step',
            'action': action,
            'step': self.current_step
        }
        
        try:
            response = self._send_message(message)
            
            if response.get('status') != 'success':
                raise RuntimeError(f"Step failed: {response.get('message')}")
            
            # Extract observation, reward, done
            observation = self._extract_observation(response)
            reward = self._calculate_reward(response)
            done = response.get('done', False)
            
            self.total_reward += reward
            self.cloudlets_completed = response.get('cloudlets_completed', 0)
            
            # Check if episode should be truncated (max steps)
            truncated = self.current_step >= self.max_steps
            terminated = done and not truncated
            
            info = {
                'step': self.current_step,
                'cloudlets_completed': self.cloudlets_completed,
                'total_reward': self.total_reward,
                'vm_selected': action,
                'response_time': response.get('response_time', 0.0),
                'waiting_time': response.get('waiting_time', 0.0)
            }
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"✗ Step error: {e}")
            # Return safe defaults
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            return observation, 0.0, True, False, {'error': str(e)}
    
    def _extract_observation(self, response: Dict[str, Any]) -> np.ndarray:
        """
        Extract observation vector from Java response.
        
        Args:
            response: Response dictionary from Java
            
        Returns:
            Observation array (num_vms * 3,)
        """
        vm_states = response.get('vm_states', [])
        
        if len(vm_states) != self.num_vms:
            print(f"⚠ Warning: Expected {self.num_vms} VM states, got {len(vm_states)}")
            # Pad or truncate to expected size
            vm_states = vm_states[:self.num_vms] + [{'cpu': 0, 'ram': 0, 'bw': 0}] * (self.num_vms - len(vm_states))
        
        # Flatten VM states into observation vector
        observation = []
        for vm in vm_states:
            observation.extend([
                vm.get('cpu', 0.0),
                vm.get('ram', 0.0),
                vm.get('bw', 0.0)
            ])
        
        return np.array(observation, dtype=np.float32)
    
    def _calculate_reward(self, response: Dict[str, Any]) -> float:
        """
        Calculate multi-objective reward from response.
        
        Reward components:
        1. Response time (negative - minimize)
        2. Load balance (positive - maximize)
        3. SLA violation (negative penalty)
        
        Args:
            response: Response dictionary from Java
            
        Returns:
            Scalar reward value
        """
        # Extract metrics
        response_time = response.get('response_time', 0.0)
        waiting_time = response.get('waiting_time', 0.0)
        sla_violation = response.get('sla_violation', 0)
        load_imbalance = response.get('load_imbalance', 0.0)
        
        # Component 1: Response time reward (negative, normalized)
        # Lower response time = higher reward
        if response_time > 0:
            time_reward = -response_time / 100.0  # Normalize by expected max time
        else:
            time_reward = 0.0
        
        # Component 2: Load balance reward (positive)
        # Lower imbalance = higher reward
        balance_reward = 1.0 - min(load_imbalance, 1.0)
        
        # Component 3: SLA violation penalty (negative)
        sla_penalty = -sla_violation * 10.0  # Heavy penalty for violations
        
        # Weighted combination
        total_reward = (
            self.reward_weights['time'] * time_reward +
            self.reward_weights['balance'] * balance_reward +
            self.reward_weights['sla'] * sla_penalty
        )
        
        return total_reward
    
    def render(self, mode='human'):
        """
        Render the environment state (optional).
        """
        if mode == 'human':
            print(f"\nEpisode: {self.episode_count}, Step: {self.current_step}")
            print(f"Total Reward: {self.total_reward:.2f}")
            print(f"Cloudlets Completed: {self.cloudlets_completed}")
    
    def close(self):
        """
        Clean up resources.
        """
        self.disconnect()
        super().close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get environment statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'episode_count': self.episode_count,
            'current_step': self.current_step,
            'total_reward': self.total_reward,
            'cloudlets_completed': self.cloudlets_completed,
            'avg_reward_per_step': self.total_reward / max(self.current_step, 1)
        }


def test_environment():
    """
    Test the CloudSim Gymnasium environment.
    """
    print("="*70)
    print("CloudSim Gymnasium Environment Test")
    print("="*70)
    
    # Create environment
    env = CloudSimEnv(num_vms=30, host='localhost', port=5555)
    
    print(f"\n✓ Environment created")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    try:
        # Test reset
        print(f"\n⏳ Testing reset...")
        observation, info = env.reset()
        print(f"✓ Reset successful")
        print(f"  Observation shape: {observation.shape}")
        print(f"  Initial observation (first 10): {observation[:10]}")
        
        # Test a few steps
        print(f"\n⏳ Testing steps...")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {i+1}: action={action}, reward={reward:.4f}, done={terminated or truncated}")
            
            if terminated or truncated:
                break
        
        print(f"\n✓ Test completed successfully!")
        print(f"\nStatistics: {env.get_statistics()}")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        print(f"\n✓ Environment closed")


if __name__ == "__main__":
    test_environment()

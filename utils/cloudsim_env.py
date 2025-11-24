import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import json


class CloudSimEnv(gym.Env):
    """
    Custom Gym Environment for CloudSim Plus Task Scheduling
    
    State Space: [vm_load_1, ..., vm_load_n, next_cloudlet_length]
    Action Space: Discrete(num_vms) - select which VM to assign the next cloudlet
    """
    
    def __init__(self, num_vms=20, host='localhost', port=9999):
        super(CloudSimEnv, self).__init__()
        
        self.num_vms = num_vms
        self.host = host
        self.port = port
        self.sock = None
        
        # Define action and observation space
        self.action_space = spaces.Discrete(num_vms)
        
        # Observation: VM loads + next cloudlet length
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(num_vms + 1,), 
            dtype=np.float32
        )
        
        self.current_state = None
        self.episode_reward = 0
        self.episode_steps = 0
        
    def connect(self):
        """Connect to CloudSim socket server"""
        if self.sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            print(f"Connected to CloudSim server at {self.host}:{self.port}")
    
    def disconnect(self):
        """Disconnect from server"""
        if self.sock:
            self.send_command({"command": "close"})
            self.sock.close()
            self.sock = None
    
    def send_command(self, command):
        """Send command to CloudSim and receive response"""
        message = json.dumps(command) + '\n'
        self.sock.sendall(message.encode())
        
        response = self.sock.recv(4096).decode()
        return json.loads(response)
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        if self.sock is None:
            self.connect()
        
        response = self.send_command({"command": "reset"})
        
        self.current_state = np.array(response['state'], dtype=np.float32)
        self.episode_reward = 0
        self.episode_steps = 0
        
        return self.current_state, {}
    
    def step(self, action):
        """Execute one step in the environment"""
        response = self.send_command({
            "command": "step",
            "action": int(action)
        })
        
        if 'error' in response:
            raise ValueError(f"Error from CloudSim: {response['error']}")
        
        self.current_state = np.array(response['state'], dtype=np.float32)
        reward = response['reward']
        done = response['done']
        info = response.get('info', {})
        
        self.episode_reward += reward
        self.episode_steps += 1
        
        # Gymnasium uses truncated instead of just done
        terminated = done
        truncated = False
        
        if terminated:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_steps
            }
        
        return self.current_state, reward, terminated, truncated, info
    
    def close(self):
        """Close the environment"""
        self.disconnect()
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        pass

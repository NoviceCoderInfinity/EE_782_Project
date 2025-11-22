"""
Python Socket Server for CloudSim Plus RL Bridge
Phase 0: Socket-based communication between Python (Agent) and Java (Environment)

This server will be used in Phase 3 for DQN/PPO implementations.
For Phase 1, Q-Learning is implemented directly in Java.
"""

import socket
import json
import numpy as np
from typing import Dict, List, Tuple

class CloudSimBridgeServer:
    """
    Socket server that receives state from Java CloudSim and sends back actions.
    This enables Python RL agents to control Java CloudSim simulations.
    """
    
    def __init__(self, host='localhost', port=9999):
        self.host = host
        self.port = port
        self.socket = None
        self.connection = None
        
    def start(self):
        """Start the socket server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        
        print(f"Python RL Agent Server started on {self.host}:{self.port}")
        print("Waiting for CloudSim Java client to connect...")
        
        self.connection, address = self.socket.accept()
        print(f"Connected to CloudSim client at {address}")
        
    def receive_state(self) -> Dict:
        """
        Receive state from Java CloudSim
        Expected format: {"vm_loads": [0.1, 0.5, 0.8, ...], "cloudlet_info": {...}}
        """
        try:
            # Receive message length first (4 bytes)
            msg_length_bytes = self.connection.recv(4)
            if not msg_length_bytes:
                return None
            
            msg_length = int.from_bytes(msg_length_bytes, byteorder='big')
            
            # Receive the actual message
            data = b''
            while len(data) < msg_length:
                chunk = self.connection.recv(min(msg_length - len(data), 4096))
                if not chunk:
                    return None
                data += chunk
            
            # Parse JSON
            state = json.loads(data.decode('utf-8'))
            return state
            
        except Exception as e:
            print(f"Error receiving state: {e}")
            return None
    
    def send_action(self, action: int):
        """
        Send action (VM ID) to Java CloudSim
        """
        try:
            # Send action as JSON
            action_data = json.dumps({"vm_id": action})
            action_bytes = action_data.encode('utf-8')
            
            # Send length first, then data
            self.connection.sendall(len(action_bytes).to_bytes(4, byteorder='big'))
            self.connection.sendall(action_bytes)
            
        except Exception as e:
            print(f"Error sending action: {e}")
    
    def close(self):
        """Close the connection"""
        if self.connection:
            self.connection.close()
        if self.socket:
            self.socket.close()
        print("Server closed")


def example_usage():
    """
    Example: Simple random agent that communicates with CloudSim
    """
    server = CloudSimBridgeServer()
    server.start()
    
    try:
        while True:
            # Receive state from CloudSim
            state = server.receive_state()
            
            if state is None:
                print("Connection closed by client")
                break
            
            print(f"Received state: {state}")
            
            # Simple random action (for demo purposes)
            vm_loads = state.get('vm_loads', [])
            num_vms = len(vm_loads)
            
            if num_vms > 0:
                # Choose VM with minimum load
                action = int(np.argmin(vm_loads))
                print(f"Sending action: VM {action}")
                server.send_action(action)
            else:
                print("No VMs available")
                break
                
    except KeyboardInterrupt:
        print("\nServer interrupted by user")
    finally:
        server.close()


if __name__ == "__main__":
    print("="*60)
    print("CloudSim Plus - Python RL Bridge Server")
    print("This will be used in Phase 3 for DQN/PPO implementations")
    print("="*60)
    print()
    
    example_usage()

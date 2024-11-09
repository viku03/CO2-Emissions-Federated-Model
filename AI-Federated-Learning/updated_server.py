import socket
import pickle
import torch
import torch.nn as nn
from typing import Dict, List
import threading
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from model import EmissionsModel
import os
import matplotlib
matplotlib.use('Agg')

class FederatedServer:
    def __init__(self, host='localhost', port=8080, num_clients=3):
        self.host = host
        self.port = port
        self.input_dim = 2781  # Make sure this matches your model's input dimension
        self.global_model = EmissionsModel(self.input_dim)
        self.clients = {}  # Changed to dict to track clients by ID
        self.current_round = 0
        self.max_rounds = 10
        self.num_clients = num_clients
        self.received_updates = defaultdict(list)
        self.lock = threading.Lock()
        self.ready_to_start = threading.Event()
        
        # Metrics tracking
        self.global_train_losses = []
        self.global_val_losses = []
        self.client_metrics = defaultdict(lambda: {'train_losses': [], 'val_losses': []})

        # Ensure output directories exist
        os.makedirs('GlobalGraph', exist_ok=True)
    
    def aggregate_models(self, updates: List[Dict], client_weights: Dict[int, float]) -> None:
        """Aggregate model updates using weighted FedAvg with type checking"""
        averaged_dict = {}
        total_weight = sum(client_weights.values())
        
        # Initialize averaged_dict with zeros of the correct type
        for key, param in self.global_model.state_dict().items():
            averaged_dict[key] = torch.zeros_like(param, dtype=param.dtype)
        
        # Weighted sum of parameters
        for update, client_id in zip(updates, client_weights.keys()):
            weight = client_weights[client_id] / total_weight
            for key in averaged_dict.keys():
                # Ensure correct dtype for multiplication
                weight_tensor = torch.tensor(weight, dtype=update[key].dtype)
                averaged_dict[key] += weight_tensor * update[key]
        
        self.global_model.load_state_dict(averaged_dict)
    
    def plot_global_metrics(self):
        """Plot global training and validation metrics"""
        plt.figure(figsize=(12, 6))
        
        # Plot global losses
        plt.subplot(1, 2, 1)
        plt.plot(self.global_train_losses, label='Global Training Loss', marker='o')
        plt.plot(self.global_val_losses, label='Global Validation Loss', marker='s')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.title('Global Model Learning Curves')
        plt.legend()
        plt.grid(True)
        
        # Plot client performances
        plt.subplot(1, 2, 2)
        for client_id, metrics in self.client_metrics.items():
            plt.plot(metrics['val_losses'], 
                    label=f'Client {client_id}', 
                    marker='o',
                    linestyle='--')
        plt.xlabel('Round')
        plt.ylabel('Validation Loss')
        plt.title('Client-wise Performance')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'GlobalGraph/global_metrics_{timestamp}.png')
        plt.close()
    
    def track_metrics(self, client_id: int, train_loss: float, val_loss: float) -> None:
        """Track metrics for each client and global model"""
        self.client_metrics[client_id]['train_losses'].append(train_loss)
        self.client_metrics[client_id]['val_losses'].append(val_loss)
        
        # Calculate and store global averages
        avg_train_loss = np.mean([metrics['train_losses'][-1] 
                                for metrics in self.client_metrics.values()])
        avg_val_loss = np.mean([metrics['val_losses'][-1] 
                              for metrics in self.client_metrics.values()])
        
        self.global_train_losses.append(avg_train_loss)
        self.global_val_losses.append(avg_val_loss)
        
        print(f"\nGlobal Metrics for Round {self.current_round + 1}:")
        print(f"Average Training Loss: {avg_train_loss:.6f}")
        print(f"Average Validation Loss: {avg_val_loss:.6f}")
    
    def handle_client(self, client_socket, address):
        """Handle individual client connections"""
        with self.lock:
            client_id = len(self.clients)
            self.clients[client_id] = client_socket
            print(f"New connection from {address} (Client {client_id})")

            if len(self.clients) == self.num_clients:
                print("All clients connected. Starting training...")
                self.ready_to_start.set()

        self.ready_to_start.wait()
        
        try:
            while self.current_round < self.max_rounds:
                # Send current model to client
                client_socket.send(pickle.dumps(self.global_model.state_dict()))
                
                # Receive client update
                data = self.receive_data(client_socket)
                if not data:
                    print(f"Client {client_id} disconnected")
                    break

                model_state = data['model_state']
                metrics = data['metrics']
                train_loss = metrics['train_loss']
                val_loss = metrics['val_loss']
                
                with self.lock:
                    self.track_metrics(client_id, train_loss, val_loss)
                    self.received_updates[self.current_round].append((model_state, client_id))
                    
                    if len(self.received_updates[self.current_round]) == len(self.clients):
                        print(f"\nRound {self.current_round + 1} complete")
                        updates, client_ids = zip(*self.received_updates[self.current_round])
                        client_weights = {cid: 1.0 for cid in client_ids}
                        
                        self.aggregate_models(updates, client_weights)
                        self.plot_global_metrics()  # Plot after each round
                        
                        self.current_round += 1
                        self.received_updates[self.current_round] = []
        
        except Exception as e:
            print(f"Error handling client {address}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with self.lock:
                if client_id in self.clients:
                    del self.clients[client_id]
            client_socket.close()

    def receive_data(self, client_socket):
        """Receive data from client with proper buffering"""
        data = b""
        while True:
            try:
                packet = client_socket.recv(4096)
                if not packet:
                    return None
                data += packet
                if len(packet) < 4096:
                    break
            except socket.error as e:
                print(f"Socket error while receiving data: {e}")
                return None

        try:
            return pickle.loads(data)
        except Exception as e:
            print(f"Error unpickling data: {e}")
            return None

    def start(self):
        """Start the federated learning server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind((self.host, self.port))
            server_socket.listen(self.num_clients)
            
            print(f"Server listening on {self.host}:{self.port}")
            print(f"Waiting for {self.num_clients} clients to connect...")
            
            while True:
                try:
                    client_socket, address = server_socket.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                    if self.current_round >= self.max_rounds:
                        print("Maximum rounds reached. Stopping server...")
                        break
                        
                except socket.error as e:
                    print(f"Socket error while accepting connection: {e}")
                    continue
                except Exception as e:
                    print(f"Unexpected error while accepting connection: {e}")
                    continue
            
            # Plot final metrics
            if self.global_train_losses:
                self.plot_global_metrics()
                print("\nFinal Global Metrics:")
                print(f"Average Training Loss: {self.global_train_losses[-1]:.6f}")
                print(f"Average Validation Loss: {self.global_val_losses[-1]:.6f}")
            
        except KeyboardInterrupt:
            print("\nServer shutting down...")
        finally:
            server_socket.close()
            print("Server stopped.")

if __name__ == "__main__":
    server = FederatedServer(num_clients=3)
    server.start()
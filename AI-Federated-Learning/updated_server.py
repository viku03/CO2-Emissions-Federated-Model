import socket
import pickle
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import threading
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os
from updated_model import EmissionsModel
import matplotlib
matplotlib.use('Agg')

class FederatedServer:
    def __init__(self, 
                 host: str = 'localhost', 
                 port: int = 8080, 
                 num_clients: int = 3,
                 input_dim: int = 2781,
                 hidden_dims: List[int] = [256, 128, 64],
                 max_rounds: int = 10,
                 early_stopping_patience: int = 3,
                 min_improvement: float = 0.001):
        """
        Initialize the Federated Learning Server
        
        Args:
            host (str): Server host address
            port (int): Server port number
            num_clients (int): Number of expected clients
            input_dim (int): Input dimension for the model
            hidden_dims (List[int]): Hidden layer dimensions
            max_rounds (int): Maximum number of federation rounds
            early_stopping_patience (int): Rounds without improvement before early stopping
            min_improvement (float): Minimum improvement threshold for global model
        """
        # Network settings
        self.host = host
        self.port = port
        self.num_clients = num_clients
        
        # Model settings
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.global_model = EmissionsModel(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=0.3,
            batch_norm_momentum=0.1
        )
        
        # Training settings
        self.max_rounds = max_rounds
        self.early_stopping_patience = early_stopping_patience
        self.min_improvement = min_improvement
        self.current_round = 0
        self.rounds_without_improvement = 0
        
        # Client management
        self.clients = {}  # Dict to track clients by ID
        self.received_updates = defaultdict(list)
        self.client_performances = defaultdict(list)
        
        # Threading controls
        self.lock = threading.Lock()
        self.ready_to_start = threading.Event()
        
        # Validation data and metrics
        self.validation_features = None
        self.validation_targets = None
        self.best_global_val_loss = float('inf')
        self.best_global_model_state = None
        
        # Metrics tracking
        self.global_train_losses = []
        self.global_val_losses = []
        self.client_metrics = defaultdict(lambda: {'train_losses': [], 'val_losses': []})
        
        # Ensure output directories exist
        os.makedirs('GlobalModels', exist_ok=True)
        os.makedirs('GlobalGraph', exist_ok=True)
        os.makedirs('Metrics', exist_ok=True)
    
    def load_validation_data(self, features: np.ndarray, targets: np.ndarray) -> None:
        """
        Load validation dataset for global model evaluation
        
        Args:
            features (np.ndarray): Validation features
            targets (np.ndarray): Validation targets
        """
        self.validation_features = features
        self.validation_targets = targets
        print(f"Validation dataset loaded with {len(features)} samples")
    
    def evaluate_global_model(self) -> float:
        """
        Evaluate current global model on validation dataset
        
        Returns:
            float: Validation loss
        """
        if self.validation_features is None or self.validation_targets is None:
            raise ValueError("Validation dataset not loaded")
            
        self.global_model.eval()
        with torch.no_grad():
            val_features = torch.FloatTensor(self.validation_features)
            val_targets = torch.FloatTensor(self.validation_targets)
            outputs = self.global_model(val_features)
            loss = nn.MSELoss()(outputs, val_targets)
        self.global_model.train()
        return loss.item()
    
    def aggregate_models(self, updates: List[Dict], client_weights: Dict[int, float]) -> bool:
        """Modified aggregation logic with better handling of model updates"""
        averaged_dict = {}
        total_weight = sum(client_weights.values())
        
        # Initialize averaged_dict with zeros
        for key, param in self.global_model.state_dict().items():
            averaged_dict[key] = torch.zeros_like(param, dtype=param.dtype)
        
        # Weighted sum of parameters
        for update, client_id in zip(updates, client_weights.keys()):
            weight = client_weights[client_id] / total_weight
            for key in averaged_dict.keys():
                weight_tensor = torch.tensor(weight, dtype=update[key].dtype)
                averaged_dict[key] += weight_tensor * update[key]
        
        # Load averaged model and evaluate
        self.global_model.load_state_dict(averaged_dict)
        new_val_loss = self.evaluate_global_model()
        
        # Check if improvement is significant
        improvement = self.best_global_val_loss - new_val_loss
        print(f"\nCurrent validation loss: {new_val_loss:.6f}")
        print(f"Previous best validation loss: {self.best_global_val_loss:.6f}")
        print(f"Improvement: {improvement:.6f}")
        
        if improvement > self.min_improvement:
            self.best_global_val_loss = new_val_loss
            self.best_global_model_state = averaged_dict.copy()  # Make a copy of the state dict
            self.rounds_without_improvement = 0
            print(f"Global model improved! New validation loss: {new_val_loss:.6f}")
            return True
        else:
            # Important: Load back the best model state if no improvement
            if self.best_global_model_state is not None:
                self.global_model.load_state_dict(self.best_global_model_state)
            self.rounds_without_improvement += 1
            print(f"No significant improvement. Keeping previous best model.")
            print(f"Rounds without improvement: {self.rounds_without_improvement}")
            return False
    
    def save_metrics(self) -> None:
        """Save training metrics to file"""
        metrics = {
            'global_train_losses': self.global_train_losses,
            'global_val_losses': self.global_val_losses,
            'client_metrics': dict(self.client_metrics),
            'best_validation_loss': self.best_global_val_loss,
            'final_round': self.current_round
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'Metrics/training_metrics_{timestamp}.pkl', 'wb') as f:
            pickle.dump(metrics, f)
    
    def plot_global_metrics(self) -> None:
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
        """
        Track metrics for each client and global model
        
        Args:
            client_id (int): Client identifier
            train_loss (float): Training loss from client
            val_loss (float): Validation loss from client
        """
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
    
    def receive_data(self, client_socket: socket.socket) -> Optional[Dict]:
        """
        Receive data from client with proper buffering
        
        Args:
            client_socket (socket.socket): Client socket connection
            
        Returns:
            Optional[Dict]: Received data or None if error
        """
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
    
    def handle_client(self, client_socket: socket.socket, address: Tuple[str, int]) -> None:
        """Modified client handling with improved round progression logic"""
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
                if self.rounds_without_improvement >= self.early_stopping_patience:
                    print("Early stopping triggered - no improvement in global model")
                    break
                
                print(f"\nStarting round {self.current_round + 1}")
                # Send current model to client
                client_socket.send(pickle.dumps(self.global_model.state_dict()))
                
                # Receive client update
                data = self.receive_data(client_socket)
                if not data:
                    print(f"Client {client_id} disconnected")
                    break

                model_state = data['model_state']
                metrics = data['metrics']
                
                with self.lock:
                    self.track_metrics(client_id, metrics['train_loss'], metrics['val_loss'])
                    self.received_updates[self.current_round].append((model_state, client_id))
                    
                    if len(self.received_updates[self.current_round]) == len(self.clients):
                        print(f"\nAll clients reported for round {self.current_round + 1}")
                        updates, client_ids = zip(*self.received_updates[self.current_round])
                        client_weights = {cid: 1.0 for cid in client_ids}
                        
                        # Aggregate models and update round counter
                        model_improved = self.aggregate_models(updates, client_weights)
                        
                        # Always increment round counter and clear updates
                        self.current_round += 1
                        self.received_updates[self.current_round] = []
                        
                        if model_improved:
                            # Save best global model
                            torch.save(self.best_global_model_state, 
                                     f'GlobalModels/global_model_round_{self.current_round}.pth')
                        
                        self.plot_global_metrics()
                        
                        print(f"Moving to round {self.current_round + 1}")
        
        except Exception as e:
            print(f"Error handling client {address}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with self.lock:
                if client_id in self.clients:
                    del self.clients[client_id]
            client_socket.close()

    
    def start(self) -> None:
        """Start the federated learning server"""
        if self.validation_features is None or self.validation_targets is None:
            raise ValueError("Please load validation data before starting the server")
            
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
            
            # Save final metrics and model
            if self.global_train_losses:
                self.save_metrics()
                self.plot_global_metrics()
                if self.best_global_model_state is not None:
                    torch.save(self.best_global_model_state, 
                             'GlobalModels/final_global_model.pth')
                print("\nFinal Global Metrics:")
                print(f"Best Validation Loss: {self.best_global_val_loss:.6f}")
            
        except KeyboardInterrupt:
            print("\nServer shutting down...")
        finally:
            server_socket.close()
            print("Server stopped.")

if __name__ == "__main__":
    # Example usage
    server = FederatedServer(
        num_clients=3,
        input_dim=2781,
        hidden_dims=[256, 128, 64],
        max_rounds=10,
        early_stopping_patience=3
    )
    
    # Load validation data before starting
    # val_features and val_targets should be numpy arrays
    # server
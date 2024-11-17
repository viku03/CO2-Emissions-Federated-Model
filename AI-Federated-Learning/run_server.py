# Save this as run_server.py
import numpy as np
from updated_server import FederatedServer

if __name__ == "__main__":
    # Create sample validation data (replace with your actual validation data)
    n_samples = 1000
    n_features = 2781  # Match the input dimension specified in server init
    
    # Generate sample validation data
    np.random.seed(42)  # For reproducibility
    val_features = np.random.randn(n_samples, n_features)
    val_targets = np.random.randn(n_samples, 1)
    
    # Initialize server
    server = FederatedServer(
        host='localhost',
        port=8080,
        num_clients=3,
        input_dim=2781,
        hidden_dims=[256, 128, 64],
        max_rounds=10,
        early_stopping_patience=3,
        min_improvement=0.001
    )
    
    # Load validation data
    server.load_validation_data(val_features, val_targets)
    
    # Start the server
    print("Starting Federated Learning Server...")
    server.start()
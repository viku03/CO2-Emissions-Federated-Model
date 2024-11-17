import flwr as fl
import torch
from updated_model import EmissionsModel

# Define a custom strategy for federated averaging
class EmissionsStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)
        # Custom logging or early stopping logic could go here
        return aggregated_parameters

# Initialize and start the server
def main():
    # Server strategy
    server_config = fl.server.ServerConfig(num_rounds=10)
    
    # Start the Flower server with the specified configuration
    fl.server.start_server(
        server_address="localhost:8080",
        config=server_config,
        strategy=fl.server.strategy.FedAvg()
    )

if __name__ == "__main__":
    main()

import flwr as fl
from typing import List, Tuple
from flwr.common import Metrics, NDArrays, Parameters, FitIns, EvaluateIns
import torch
import torch.nn as nn
import numpy as np

class CO2Model(nn.Module):
    def __init__(self):
        super(CO2Model, self).__init__()
        self.input_size = 2
        self.hidden1_size = 128
        self.hidden2_size = 64
        self.hidden3_size = 32
        self.output_size = 1

        # Layer definitions
        self.fc1 = nn.Linear(self.input_size, self.hidden1_size)
        self.fc2 = nn.Linear(self.hidden1_size, self.hidden2_size)
        self.fc3 = nn.Linear(self.hidden2_size, self.hidden3_size)
        self.fc4 = nn.Linear(self.hidden3_size, self.output_size)
        
        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm1d(self.hidden1_size, track_running_stats=False)
        self.batch_norm2 = nn.BatchNorm1d(self.hidden2_size, track_running_stats=False)
        self.batch_norm3 = nn.BatchNorm1d(self.hidden3_size, track_running_stats=False)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        if x.size(0) > 1:  # Only apply batch norm if batch size > 1
            x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        if x.size(0) > 1:
            x = self.batch_norm3(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = CO2Model()
        
        # Initialize with eval mode
        self.model.eval()
        
        # Store parameter names for proper aggregation
        self.param_names = [name for name, _ in self.model.named_parameters()]

    def initialize_parameters(self, client_manager=None) -> Parameters:
        """Initialize model parameters."""
        params = []
        for param in self.model.parameters():
            params.append(param.detach().cpu().numpy())
        return fl.common.ndarrays_to_parameters(params)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[NDArrays, int]],
        failures: List[BaseException],
    ) -> Tuple[NDArrays, dict]:
        """Aggregate model parameters with proper parameter tracking."""
        if not results:
            return None, {}

        # Aggregate parameters using parent class method
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)[0]
        
        if aggregated_parameters is not None:
            # Convert parameters back to tensors
            parameters_tensors = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            # Update model parameters
            with torch.no_grad():
                for param_tensor, name in zip(parameters_tensors, self.param_names):
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            param.data = torch.tensor(param_tensor)
                            break

        return aggregated_parameters, {}

    def aggregate_evaluate_metrics(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """Aggregate evaluation metrics from clients."""
        if not metrics:
            return {}
            
        aggregated = {}
        for metric_name in ["loss", "r2"]:
            metric_values = [m[1][metric_name] for m in metrics if metric_name in m[1]]
            if metric_values:
                aggregated[metric_name] = sum(metric_values) / len(metric_values)
            
        return aggregated

def main():
    # Server configuration
    server_config = fl.server.ServerConfig(num_rounds=3)
    
    # Initialize strategy with proper configuration
    strategy = CustomFedAvg(
        min_available_clients=1,
        min_fit_clients=1,
        min_evaluate_clients=1,
    )
    
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=server_config,
        strategy=strategy
    )

if __name__ == "__main__":
    main()
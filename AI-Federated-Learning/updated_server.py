import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, Parameters
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the ImprovedCO2Model from client.py
from updated_client import ImprovedCO2Model

class ImprovedFedAvg(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *args,
        min_available_clients: int = 3,
        fraction_fit: float = 0.8,
        fraction_evaluate: float = 0.5,
        min_fit_clients: int = 3,
        min_evaluate_clients: int = 2,
        **kwargs,
    ):
        super().__init__(
            *args,
            min_available_clients=min_available_clients,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            **kwargs,
        )
        
        self.model = ImprovedCO2Model()
        self.model.eval()
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(f"logs/server/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        
        # Track best model performance
        self.best_accuracy = float('-inf')
        self.best_parameters = None
        
    def initialize_parameters(self, client_manager=None) -> Parameters:
        """Initialize model parameters."""
        params = []
        for param in self.model.parameters():
            params.append(param.detach().cpu().numpy())
        return fl.common.ndarrays_to_parameters(params)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Parameters, Dict]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict]:
        """Aggregate model parameters and training metrics."""
        if not results:
            return None, {}
        
        # Extract parameters and metrics from results
        parameters_list = [parameters for parameters, _ in results]
        metrics_list = [metrics for _, metrics in results]
        
        # Aggregate parameters (weighted by number of examples)
        weights = [metrics["num_examples"] for _, metrics in results]
        total_examples = sum(weights)
        weighted_metrics = {
            "loss": np.average(
                [metrics["loss"] for metrics in metrics_list],
                weights=weights
            ),
            "rmse": np.average(
                [metrics["rmse"] for metrics in metrics_list],
                weights=weights
            ),
            "mae": np.average(
                [metrics["mae"] for metrics in metrics_list],
                weights=weights
            )
        }
        
        # Log metrics to TensorBoard
        self.writer.add_scalar('Train/Loss', weighted_metrics["loss"], server_round)
        self.writer.add_scalar('Train/RMSE', weighted_metrics["rmse"], server_round)
        self.writer.add_scalar('Train/MAE', weighted_metrics["mae"], server_round)
        
        # Aggregate parameters
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)[0]
        
        return aggregated_parameters, weighted_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[Parameters, Dict]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict]:
        """Aggregate evaluation metrics."""
        if not results:
            return None, {}
        
        # Extract metrics
        metrics_list = [metrics for _, metrics in results]
        
        # Calculate weighted averages
        weights = [metrics["num_examples"] for _, metrics in results]
        weighted_metrics = {
            "loss": np.average(
                [metrics["loss"] for metrics in metrics_list],
                weights=weights
            ),
            "rmse": np.average(
                [metrics["rmse"] for metrics in metrics_list],
                weights=weights
            ),
            "mae": np.average(
                [metrics["mae"] for metrics in metrics_list],
                weights=weights
            )
        }
        
        # Log evaluation metrics to TensorBoard
        self.writer.add_scalar('Eval/Loss', weighted_metrics["loss"], server_round)
        self.writer.add_scalar('Eval/RMSE', weighted_metrics["rmse"], server_round)
        self.writer.add_scalar('Eval/MAE', weighted_metrics["mae"], server_round)
        
        # Update best model if current model performs better
        if weighted_metrics["rmse"] < self.best_accuracy:
            self.best_accuracy = weighted_metrics["rmse"]
            # Get parameters from the first result (they should all be the same)
            self.best_parameters = results[0][0]
            
            # Save best model parameters
            torch.save(
                self.model.state_dict(),
                f"models/best_model_round_{server_round}.pth"
            )
        
        return weighted_metrics["loss"], weighted_metrics

def main():
    try:
        # Create necessary directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs/server", exist_ok=True)
        
        logger.info("Starting Federated Learning server...")
        
        # Define strategy with proper logging
        logger.info("Initializing strategy...")
        strategy = ImprovedFedAvg(
            min_available_clients=1,  # Reduced for testing
            min_fit_clients=1,
            min_evaluate_clients=1,
            fraction_fit=0.8,
            fraction_evaluate=0.5,
        )
        
        # Configure server
        logger.info("Configuring server...")
        config = fl.server.ServerConfig(num_rounds=50)
        
        # Start server with proper error handling
        logger.info("Starting server on [::]:8081")
        fl.server.start_server(
            server_address="[::]:8081",  # Using IPv6 notation
            strategy=strategy,
            config=config,
        )
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
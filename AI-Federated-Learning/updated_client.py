import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedCO2Model(nn.Module):
    def __init__(self):
        super(ImprovedCO2Model, self).__init__()
        self.input_size = 2
        self.hidden_sizes = [256, 128, 64, 32]
        
        # Layer definitions with skip connections
        self.layers = nn.ModuleList()
        prev_size = self.input_size
        
        for hidden_size in self.hidden_sizes:
            self.layers.append(nn.Sequential(
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size, track_running_stats=False),
                nn.ReLU(),
                nn.Dropout(0.3)
            ))
            prev_size = hidden_size
            
        self.output_layer = nn.Linear(prev_size, 1)
        
        # Skip connection layers
        self.skip_layers = nn.ModuleList([
            nn.Linear(self.input_size, size) 
            for size in self.hidden_sizes
        ])
        
    def forward(self, x):
        original_input = x
        
        for i, (layer, skip) in enumerate(zip(self.layers, self.skip_layers)):
            if x.size(0) > 1:  # Batch size check for BatchNorm
                skip_connection = skip(original_input)
                x = layer(x) + skip_connection
            else:
                x = layer(x)
                
        return self.output_layer(x)

class ImprovedEmissionClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, val_data, client_id):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Loss functions
        self.criterion = nn.HuberLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-4,
            amsgrad=True
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # TensorBoard setup
        log_dir = f"logs/client_{client_id}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(log_dir)
        
        # Early stopping parameters
        self.best_val_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
        
    def get_parameters(self, config):
        return [param.cpu().detach().numpy() for name, param in self.model.named_parameters()]
    
    def set_parameters(self, parameters):
        params_dict = zip([name for name, _ in self.model.named_parameters()], parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log to TensorBoard
            step = epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), step)
            
        return total_loss / len(train_loader)
    
    def evaluate_loader(self, val_loader):
        """Internal evaluation method for training"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        # Calculate metrics
        mse = np.mean((np.array(all_predictions) - np.array(all_targets)) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
        
        return {
            'loss': total_loss / len(val_loader),
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
    
    def fit(self, parameters, config):
        logger.info("Starting fit process")
        self.set_parameters(parameters)
        
        # Create k-fold cross-validation splits
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Training parameters
        epochs = 5
        batch_size = 32
        
        X_train, y_train = self.train_data
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            logger.info(f"Training Fold {fold+1}")
            
            # Create data loaders for this fold
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1, 1)),
                batch_size=batch_size,
                sampler=train_sampler
            )
            
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1, 1)),
                batch_size=batch_size,
                sampler=val_sampler
            )
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                train_loss = self.train_epoch(train_loader, epoch)
                
                # Validation
                val_metrics = self.evaluate_loader(val_loader)
                val_loss = val_metrics['loss']
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Log metrics
                self.writer.add_scalar(f'Fold_{fold}/Train/Loss', train_loss, epoch)
                self.writer.add_scalar(f'Fold_{fold}/Val/Loss', val_loss, epoch)
                self.writer.add_scalar(f'Fold_{fold}/Val/RMSE', val_metrics['rmse'], epoch)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping triggered in fold {fold+1}")
                        break
                
                logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            fold_results.append(val_metrics)
        
        # Average results across folds
        avg_metrics = {
            metric: np.mean([fold[metric] for fold in fold_results])
            for metric in fold_results[0].keys()
        }
        
        # Log final averaged metrics
        for metric, value in avg_metrics.items():
            self.writer.add_scalar(f'Final/{metric}', value)
        
        return self.get_parameters({}), len(X_train), avg_metrics
    
    def evaluate(self, parameters, config):
        """Evaluation method required by Flower's NumPyClient interface"""
        logger.info("Starting evaluation")
        self.set_parameters(parameters)
        
        X_val, y_val = self.val_data
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).reshape(-1, 1)),
            batch_size=32
        )
        
        metrics = self.evaluate_loader(val_loader)
        return float(metrics['loss']), len(X_val), metrics

def main():
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data_path = "Dataset/flight.csv"
        
        if not os.path.exists(data_path):
            logger.error(f"Dataset not found at {data_path}")
            sys.exit(1)
            
        data = pd.read_csv(data_path)
        
        features = ['LATITUDE', 'LONGITUDE', 'GHG QUANTITY (METRIC TONS CO2e)']
        data_clean = data[features].copy()
        data_clean = data_clean.dropna()
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data_clean)
        data_scaled = pd.DataFrame(scaled_features, columns=data_clean.columns)
        
        X = data_scaled[['LATITUDE', 'LONGITUDE']].values
        y = data_scaled['GHG QUANTITY (METRIC TONS CO2e)'].values
        
        # Split data for this client
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        logger.info("Data preprocessing completed")
        
        # Initialize model and client
        logger.info("Initializing model and client...")
        model = ImprovedCO2Model()
        client = ImprovedEmissionClient(
            model=model,
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            client_id=1
        )
        
        # Start client
        logger.info("Starting client...")
        fl.client.start_client(
            server_address="[::]:8081",
            client=client.to_client(),
        )
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
import socket
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import seaborn as sns
from datetime import datetime

from model import EmissionsModel

class EmissionsDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class TrainingTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.actuals = []
        
    def update(self, train_loss, val_loss, predictions, actuals):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.predictions.extend(predictions)
        self.actuals.extend(actuals)
        
    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.legend()
        plt.grid(True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'LossGraph/loss_plot_{timestamp}.png')
        plt.close()
        
    def plot_prediction_vs_actual(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.actuals, self.predictions, alpha=0.5)
        plt.plot([min(self.actuals), max(self.actuals)], 
                [min(self.actuals), max(self.actuals)], 
                'r--', label='Perfect Prediction')
        plt.xlabel('Actual Emissions')
        plt.ylabel('Predicted Emissions')
        plt.title('Predicted vs Actual Emissions')
        plt.legend()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'ValidationGraph/prediction_plot_{timestamp}.png')
        plt.close()

class FederatedClient:
    def __init__(self, year_data: pd.DataFrame, host='localhost', port=8080):
        self.data = year_data
        self.host = host
        self.port = port
        
        # Initialize scalers and encoder
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.company_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Calculate input dimension after feature engineering
        sample_features = self.preprocess_data(training=True)[0]
        self.input_dim = sample_features.shape[1]
        
        # Initialize model and tracking
        self.model = EmissionsModel(self.input_dim)
        self.tracker = TrainingTracker()
    
    def receive_data(self, client_socket):
        """Receive data from server with proper buffering"""
        data = b""
        while True:
            packet = client_socket.recv(4096)
            if not packet:
                return None
            data += packet
            if len(packet) < 4096:
                break
        return pickle.loads(data)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from the data"""
        df = df.copy()
        
        # Extract month from reporting year (assuming it's relevant)
        df['month'] = pd.to_datetime(df['REPORTING YEAR'].astype(str), format='%Y').dt.month
        
        # Calculate distance from mean facility location
        mean_lat = df['LATITUDE'].mean()
        mean_lon = df['LONGITUDE'].mean()
        df['distance_from_mean'] = np.sqrt(
            (df['LATITUDE'] - mean_lat)**2 + 
            (df['LONGITUDE'] - mean_lon)**2
        )
        
        # Create features for facility density in area
        df['facility_density'] = df.groupby('ZIP CODE')['FACILITY NAME'].transform('count')
        
        return df
    
    def preprocess_data(self, training=True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare the data for training or prediction"""
        # Engineer features
        df = self.engineer_features(self.data)
        
        # Prepare feature matrices
        numeric_features = df[['LATITUDE', 'LONGITUDE', 'month', 
                             'distance_from_mean', 'facility_density']].values
        
        # One-hot encode company names
        company_features = df['PARENT COMPANIES'].values.reshape(-1, 1)
        
        if training:
            company_encoded = self.company_encoder.fit_transform(company_features)
        else:
            company_encoded = self.company_encoder.transform(company_features)
        
        # Combine all features
        features = np.hstack([
            numeric_features,
            company_encoded
        ])
        
        # Scale features
        if training:
            features = self.feature_scaler.fit_transform(features)
        else:
            features = self.feature_scaler.transform(features)
        
        # Prepare targets
        if 'GHG QUANTITY (METRIC TONS CO2e)' in df.columns:
            targets = df['GHG QUANTITY (METRIC TONS CO2e)'].values.reshape(-1, 1)
            if training:
                targets = self.target_scaler.fit_transform(targets)
            else:
                targets = self.target_scaler.transform(targets)
        else:
            targets = None
            
        return features, targets
    
    def create_data_loaders(self, features, targets, batch_size=32, val_split=0.2):
        """Create training and validation data loaders"""
        dataset = EmissionsDataset(features, targets)
        
        # Split dataset into training and validation
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        return train_loader, val_loader
    
    def validate(self, val_loader, criterion) -> Tuple[float, list, list]:
        """Run validation and return loss and predictions"""
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = self.model(features)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                # Store predictions and actuals for plotting
                pred = self.target_scaler.inverse_transform(outputs.numpy())
                true = self.target_scaler.inverse_transform(targets.numpy())
                predictions.extend(pred.flatten().tolist())
                actuals.extend(true.flatten().tolist())
        
        return total_loss / len(val_loader), predictions, actuals
    

    def train_local_model(self, epochs=50, batch_size=32, learning_rate=0.001):
        """Train the model with early stopping and validation"""
        features, targets = self.preprocess_data(training=True)
        train_loader, val_loader = self.create_data_loaders(features, targets, batch_size)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        early_stopping = EarlyStopping(patience=5)
        
        best_val_loss = float('inf')
        best_train_loss = None
        
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            
            for batch_features, batch_targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_train_loss += loss.item()
            
            # Calculate average training loss
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation
            avg_val_loss, predictions, actuals = self.validate(val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Update tracker
            self.tracker.update(avg_train_loss, avg_val_loss, predictions, actuals)
            
            print(f"Epoch {epoch + 1}")
            print(f"Training Loss: {avg_train_loss:.6f}")
            print(f"Validation Loss: {avg_val_loss:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_train_loss = avg_train_loss
            
            # Early stopping check
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        # Plot training results
        self.tracker.plot_losses()
        self.tracker.plot_prediction_vs_actual()
        
        return best_train_loss, best_val_loss

    def start_training(self):
        """Connect to server and participate in federated learning"""
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            client_socket.connect((self.host, self.port))
            print("Connected to server")
            
            while True:
                # Receive global model from server
                model_data = self.receive_data(client_socket)
                if not model_data:
                    break
                
                # Update local model with global parameters
                self.model.load_state_dict(model_data)
                
                # Train local model and get metrics
                train_loss, val_loss = self.train_local_model()
                
                # Send updated model and metrics back to server
                update_data = {
                    'model_state': self.model.state_dict(),
                    'metrics': {
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    }
                }
                client_socket.send(pickle.dumps(update_data))
        
        except Exception as e:
            print(f"Error during training: {e}")
        finally:
            client_socket.close()

if __name__ == "__main__":
    # Example data
    data = pd.read_csv( 'Dataset/2023.csv' )
    
    client = FederatedClient(data)
    client.start_training()
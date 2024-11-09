import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict

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
        
        # Batch normalization layers with track_running_stats=False
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

class EmissionClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.device = torch.device("mps")
        
    def get_parameters(self, config):
        """Get parameters that should be serialized."""
        return [param.cpu().detach().numpy() for name, param in self.model.named_parameters()
                if 'running' not in name]  # Exclude running stats
    
    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy ndarrays."""
        params_dict = zip([name for name, _ in self.model.named_parameters() 
                         if 'running' not in name], parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        # Update only trainable parameters
        current_state = self.model.state_dict()
        for key in current_state:
            if key in state_dict:
                current_state[key] = state_dict[key]
        
        self.model.load_state_dict(current_state, strict=False)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        total_loss = 0
        
        print("\nStarting training round...")
        for epoch in range(5):
            epoch_loss = 0
            for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/5, Batch {batch_idx}/{len(self.train_loader)}, "
                          f"Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} completed - Avg Loss: {avg_epoch_loss:.4f}")
            total_loss += avg_epoch_loss
            
        return self.get_parameters({}), len(self.train_loader.dataset), {"loss": total_loss / 5}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(self.device)
            
            predictions = self.model(X_test_tensor)
            loss = self.criterion(predictions, y_test_tensor)
            
            y_mean = y_test_tensor.mean()
            total_sum_squares = ((y_test_tensor - y_mean) ** 2).sum()
            residual_sum_squares = ((y_test_tensor - predictions) ** 2).sum()
            r2 = 1 - (residual_sum_squares / total_sum_squares)
            
        return float(loss.cpu().numpy()), len(y_test), {
            "loss": float(loss.cpu().numpy()),
            "r2": float(r2.cpu().numpy())
        }

def main():
    # Load and preprocess data
    data = pd.read_csv("Dataset/2023.csv")
    
    features = ['LATITUDE', 'LONGITUDE', 'GHG QUANTITY (METRIC TONS CO2e)']
    data_clean = data[features].copy()
    data_clean = data_clean.dropna()
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data_clean)
    data_scaled = pd.DataFrame(scaled_features, columns=data_clean.columns)
    
    X = data_scaled[['LATITUDE', 'LONGITUDE']].values
    y = data_scaled['GHG QUANTITY (METRIC TONS CO2e)'].values
    
    global X_test, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    device = torch.device("mps")
    print(f"Using device: {device}")
    
    model = CO2Model()
    model = model.to(device)
    client = EmissionClient(model, train_loader)
    
    fl.client.start_client(
        server_address="0.0.0.0:8081",
        client=client
    )

if __name__ == "__main__":
    main()
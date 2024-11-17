import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the neural network model
class EmissionsModel(nn.Module):
    def __init__(self, input_dim):
        super(EmissionsModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(128, momentum=0.1)
        self.batch_norm2 = nn.BatchNorm1d(64, momentum=0.1)
        self.batch_norm3 = nn.BatchNorm1d(32, momentum=0.1)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm3(self.layer3(x)))
        x = self.layer4(x)
        return x

# Load and preprocess data
def load_data(year):
    # Load the CSV file for the specified year
    data = pd.read_csv(f"Dataset/{year}.csv")

    # Select features and target
    features = data[["LATITUDE", "LONGITUDE"]]
    target = data["GHG QUANTITY (METRIC TONS CO2e)"]

    # Scale features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Split the data into train and test sets
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Convert to PyTorch tensors
    train_features = torch.tensor(train_features, dtype=torch.float32)
    train_target = torch.tensor(train_target.values, dtype=torch.float32).view(-1, 1)
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_target = torch.tensor(test_target.values, dtype=torch.float32).view(-1, 1)

    # Create DataLoader instances
    train_loader = DataLoader(TensorDataset(train_features, train_target), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_features, test_target), batch_size=32, shuffle=False)

    return train_loader, test_loader

# Define Flower client
class CO2EmissionsClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.model.to(self.device)

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(1):  # Adjust number of epochs as needed
            for features, target in self.train_loader:
                features, target = features.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.MSELoss()
        loss = 0
        with torch.no_grad():
            for features, target in self.test_loader:
                features, target = features.to(self.device), target.to(self.device)
                outputs = self.model(features)
                loss += criterion(outputs, target).item()

        loss /= len(self.test_loader)
        return float(loss), len(self.test_loader.dataset), {}

def main():
    # Device configuration
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, test_loader = load_data(2023)

    # Initialize the model
    model = EmissionsModel(input_dim=2)  # LATITUDE and LONGITUDE

    # Create and start Flower client
    client = CO2EmissionsClient(model, train_loader, test_loader, device)
    
    # Using the new `start_client()` method (as the previous `start_numpy_client()` is deprecated)
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn

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
        # Handle small batch sizes
        if x.size(0) == 1:
            self.batch_norm1.eval()
            self.batch_norm2.eval()
            self.batch_norm3.eval()
            
        x = self.relu(self.batch_norm1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm3(self.layer3(x)))
        x = self.layer4(x)
        return x
        
    def to(self, device):
        # Override to method to ensure all parts of the model are moved to device
        model = super().to(device)
        return model
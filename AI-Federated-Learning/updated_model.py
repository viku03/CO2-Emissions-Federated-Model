import torch
import torch.nn as nn

class EmissionsModel(nn.Module):
    """
    Neural network model for CO2 emissions prediction with enhanced architecture.
    Features residual connections, batch normalization, and dropout for better stability and performance.
    
    Args:
        input_dim (int): Dimension of input features
        hidden_dims (list, optional): List of hidden layer dimensions. Defaults to [128, 64, 32]
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2
        batch_norm_momentum (float, optional): Momentum for batch normalization. Defaults to 0.1
    """
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: list = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 batch_norm_momentum: float = 0.1):
        super(EmissionsModel, self).__init__()
        
        # Save configurations
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Main layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dims[0], momentum=batch_norm_momentum))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i + 1], momentum=batch_norm_momentum))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # Residual connections
        self.residual_layers = nn.ModuleList()
        self.residual_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.residual_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        
        # Additional components
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dims[-1])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
        
        for layer in self.residual_layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)
        
        nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='relu')
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Predicted emissions value(s)
        """
        # Handle single sample batch normalization
        if x.size(0) == 1:
            for batch_norm in self.batch_norms:
                batch_norm.eval()
        
        # Initial identity for residual connection
        identity = x
        
        # Process through hidden layers with residual connections
        for i in range(len(self.layers)):
            # Main path
            x = self.layers[i](x)
            x = self.batch_norms[i](x)
            x = self.relu(x)
            x = self.dropout(x)
            
            # Residual connection
            identity = self.residual_layers[i](identity)
            
            # Add residual (if shapes match)
            if x.shape == identity.shape:
                x = x + identity
            identity = x
        
        # Final layer normalization
        x = self.layer_norm(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with the model in evaluation mode.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Model predictions
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        self.train()
        return predictions
    
    def to(self, device: torch.device) -> 'EmissionsModel':
        """
        Move the model to the specified device.
        
        Args:
            device (torch.device): Device to move the model to
            
        Returns:
            EmissionsModel: The model on the specified device
        """
        model = super().to(device)
        for batch_norm in self.batch_norms:
            batch_norm.to(device)
        return model
    
    def get_number_of_parameters(self) -> int:
        """
        Calculate the total number of trainable parameters in the model.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
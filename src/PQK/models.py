# Import necessary PyTorch modules and custom quantum layer
import torch
import torch.nn as nn
from utils import QuantumConv2D__Trainable_Type2


class Type2_TrainableKernel__Hybrid(nn.Module):
    """Hybrid quantum-classical neural network with a single quantum convolutional layer.
    
    This model uses one quantum convolutional layer followed by classical fully connected
    layers for digit classification. The quantum layer processes image patches through
    a parameterized quantum circuit.
    
    Args:
        num_classes (int): Number of output classes (default: 10 for MNIST digits)
        device (str): Device to run the model on ('cpu', 'cuda', or 'mps')
    """
    def __init__(self, num_classes=10, device='cpu'):
        super(Type2_TrainableKernel__Hybrid, self).__init__()
        
        # First quantum convolutional layer
        # Processes 28x28 grayscale images with 5x5 quantum kernels
        self.conv1 = QuantumConv2D__Trainable_Type2(
            img_height=28,        # Input image height
            img_width=28,         # Input image width  
            kernel_size=(5, 5),   # Quantum kernel size
            stride=1,             # Convolution stride
            in_channels=1,        # Grayscale input (1 channel)
            output_size_merlin=16, # Number of output feature maps
            modes_merlin=5,       # Number of quantum modes in circuit
            depth_merlin=2,       # Depth of quantum circuit layers
            layer_global_idx=1,   # Global layer identifier for parameter naming
            device=device
        )

        # Standard neural network layers
        self.relu = nn.ReLU()  # ReLU activation function
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling for downsampling
        
        # Alternative second quantum layer configurations (not used in single layer model)
        # conv2__wo_pool: For processing without pooling (26x26 input)
        self.conv2__wo_pool = QuantumConv2D__Trainable_Type2(
            img_height=26,        # Input size after first conv (28-5+1=24, but set to 26)
            img_width=26,
            kernel_size=(5, 5),
            stride=1,
            in_channels=16,       # Matches output of first quantum layer
            output_size_merlin=32,
            modes_merlin=5,
            depth_merlin=2,
            layer_global_idx=2,
            device=device
        )
        
        # conv2__w_pool: For processing after pooling (13x13 input)
        self.conv2__w_pool = QuantumConv2D__Trainable_Type2(
            img_height=13,        # Input size after pooling (24/2 = 12, but set to 13)
            img_width=13,
            kernel_size=(5, 5),
            stride=1,
            in_channels=16,
            output_size_merlin=32,
            modes_merlin=5,
            depth_merlin=2,
            layer_global_idx=2,
            device=device
        )

        # Fully connected layers for classification
        self.flatten = nn.Flatten()  # Flatten feature maps to 1D vector
        self.fc = nn.Linear(64*7*7, num_classes)  # Direct classification layer (unused)
        self.fc1__wo_pool = nn.Linear(16*24*24, 512)  # First FC layer (16 channels, 24x24 features)
        self.fc1__w_pool = nn.Linear(32*4*4, 512)     # Alternative first FC layer (unused)
        self.fc2 = nn.Linear(512, 64)                 # Second FC layer
        self.fc3 = nn.Linear(64, num_classes)         # Final classification layer

        # Dropout layers for regularization to prevent overfitting
        self.drop1 = nn.Dropout(0.2)  # 20% dropout after first FC layer
        self.drop2 = nn.Dropout(0.4)  # 40% dropout after second FC layer

    def forward(self, x):
        """Forward pass through the hybrid quantum-classical network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            tuple: (predictions, conv1_embeddings, conv1_embeddings_duplicate)
                - predictions: Class predictions of shape (batch_size, num_classes)
                - conv1_embeddings: Quantum layer outputs for analysis
                - conv1_embeddings_duplicate: Same as conv1_embeddings (for compatibility)
        """
        # Process through quantum convolutional layer
        conv1_out = self.conv1(x)  # Shape: (batch_size, 16, 24, 24)
        x = self.relu(conv1_out)   # Apply ReLU activation
        
        # Flatten for fully connected layers
        x = self.flatten(x)        # Shape: (batch_size, 16*24*24)
        
        # Pass through fully connected layers with dropout
        x = self.fc1__wo_pool(x)   # Shape: (batch_size, 512)
        x = self.drop1(x)          # Apply first dropout
        x = self.fc2(x)            # Shape: (batch_size, 64)
        x = self.drop2(x)          # Apply second dropout
        x = self.fc3(x)            # Shape: (batch_size, num_classes)

        # Return predictions and quantum layer outputs (detached for analysis)
        return x, conv1_out.clone().detach(), conv1_out.clone().detach()


class Type2_TrainableKernel__Hybrid__2_layers(nn.Module):
    """Hybrid quantum-classical neural network with two quantum convolutional layers.
    
    This model uses two sequential quantum convolutional layers followed by classical
    fully connected layers. The deeper quantum processing allows for more complex
    feature extraction from the input images.
    
    Args:
        num_classes (int): Number of output classes (default: 10 for MNIST digits)
        device (str): Device to run the model on ('cpu', 'cuda', or 'mps')
    """
    def __init__(self, num_classes=10, device='cpu'):
        super(Type2_TrainableKernel__Hybrid__2_layers, self).__init__()
        
        # First quantum convolutional layer
        # Uses smaller 3x3 kernels for finer feature extraction
        self.conv1 = QuantumConv2D__Trainable_Type2(
            img_height=28,        # Input image height
            img_width=28,         # Input image width
            kernel_size=(3, 3),   # Smaller quantum kernel size for finer features
            stride=1,             # Convolution stride
            in_channels=1,        # Grayscale input
            output_size_merlin=16, # Number of output feature maps
            modes_merlin=5,       # Number of quantum modes
            depth_merlin=2,       # Quantum circuit depth
            layer_global_idx=1,   # Layer identifier
            device=device
        )

        # Standard neural network layers
        self.relu = nn.ReLU()  # ReLU activation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling

        # Second quantum convolutional layer (without pooling)
        # Processes output from first quantum layer
        self.conv2__wo_pool = QuantumConv2D__Trainable_Type2(
            img_height=26,        # Input size after first conv (28-3+1=26)
            img_width=26,
            kernel_size=(5, 5),   # Larger kernel for broader feature capture
            stride=1,
            in_channels=16,       # Matches output channels of first layer
            output_size_merlin=32, # More output channels for richer features
            modes_merlin=5,
            depth_merlin=2,
            layer_global_idx=2,   # Unique layer identifier
            device=device
        )
        
        # Alternative second quantum layer (with pooling) - unused in this architecture
        self.conv2__w_pool = QuantumConv2D__Trainable_Type2(
            img_height=13,        # Input size after pooling
            img_width=13,
            kernel_size=(5, 5),
            stride=1,
            in_channels=16,
            output_size_merlin=32,
            modes_merlin=5,
            depth_merlin=2,
            layer_global_idx=2,
            device=device
        )

        # Fully connected layers for final classification
        self.flatten = nn.Flatten()  # Flatten 2D feature maps
        self.fc = nn.Linear(64*7*7, num_classes)    # Unused direct classification
        self.fc1__wo_pool = nn.Linear(32*22*22, 512) # First FC (32 channels, 22x22 features)
        self.fc1__w_pool = nn.Linear(32*4*4, 512)    # Alternative first FC (unused)
        self.fc2 = nn.Linear(512, 64)                # Hidden layer
        self.fc3 = nn.Linear(64, num_classes)        # Output classification layer

        # Dropout layers for regularization
        self.drop1 = nn.Dropout(0.2)  # Light dropout after first FC
        self.drop2 = nn.Dropout(0.4)  # Heavier dropout before final layer

    def forward(self, x):
        """Forward pass through the two-layer hybrid quantum-classical network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            tuple: (predictions, conv1_embeddings, conv2_embeddings)
                - predictions: Class predictions of shape (batch_size, num_classes)
                - conv1_embeddings: First quantum layer outputs
                - conv2_embeddings: Second quantum layer outputs
        """
        # First quantum convolutional layer
        conv1_out = self.conv1(x)  # Shape: (batch_size, 16, 26, 26)
        x = self.relu(conv1_out)   # Apply activation

        # Second quantum convolutional layer
        conv2_out = self.conv2__wo_pool(x)  # Shape: (batch_size, 32, 22, 22)
        x = self.relu(conv2_out)            # Apply activation

        # Flatten and pass through fully connected layers
        x = self.flatten(x)        # Shape: (batch_size, 32*22*22)
        x = self.fc1__wo_pool(x)   # Shape: (batch_size, 512)
        x = self.drop1(x)          # Apply dropout
        x = self.fc2(x)            # Shape: (batch_size, 64)
        x = self.drop2(x)          # Apply dropout
        x = self.fc3(x)            # Shape: (batch_size, num_classes)

        # Return predictions and both quantum layer outputs
        return x, conv1_out.clone().detach(), conv2_out.clone().detach()


class Type2_TrainableKernel__Hybrid__2_layers__combined_parallel(nn.Module):
    """Hybrid model with parallel classical and quantum processing branches.
    
    This advanced architecture processes the same input through both classical
    convolutional layers and quantum convolutional layers in parallel, then
    combines their features for improved classification performance.
    
    The parallel design allows the model to leverage both classical pattern
    recognition and quantum feature extraction simultaneously.
    
    Args:
        num_classes (int): Number of output classes (default: 10 for MNIST digits)
        device (str): Device to run the model on ('cpu', 'cuda', or 'mps')
    """
    def __init__(self, num_classes=10, device='cpu'):
        super(Type2_TrainableKernel__Hybrid__2_layers__combined_parallel, self).__init__()
        
        # Classical CNN branch for traditional feature extraction
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=0)   # First classical conv layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)  # Second classical conv layer
        
        # Quantum CNN branch for quantum feature extraction
        # First quantum convolutional layer
        self.conv1_PQK = QuantumConv2D__Trainable_Type2(
            img_height=28,        # Same input size as classical branch
            img_width=28,
            kernel_size=(3, 3),   # Matching kernel size with classical branch
            stride=1,
            in_channels=1,        # Grayscale input
            output_size_merlin=16, # Same output channels as classical branch
            modes_merlin=5,       # Quantum circuit modes
            depth_merlin=2,       # Quantum circuit depth
            layer_global_idx=1,   # Unique quantum layer identifier
            device=device
        )

        # Second quantum convolutional layer (without pooling)
        self.conv2__wo_pool_PQK = QuantumConv2D__Trainable_Type2(
            img_height=26,        # Input size after first quantum conv (28-3+1=26)
            img_width=26,
            kernel_size=(5, 5),   # Matching kernel size with classical branch
            stride=1,
            in_channels=16,       # Matches first quantum layer output
            output_size_merlin=32, # Same output channels as classical branch
            modes_merlin=5,
            depth_merlin=2,
            layer_global_idx=2,   # Unique quantum layer identifier
            device=device
        )
        
        # Alternative second quantum layer (with pooling) - unused in this model
        self.conv2__w_pool_PQK = QuantumConv2D__Trainable_Type2(
            img_height=13,        # Input size after pooling
            img_width=13,
            kernel_size=(5, 5),
            stride=1,
            in_channels=16,
            output_size_merlin=32,
            modes_merlin=5,
            depth_merlin=2,
            layer_global_idx=2,
            device=device
        )

        # Feature fusion layer to combine classical and quantum branches
        # Takes concatenated 64 channels (32 classical + 32 quantum) and reduces to 32
        self.conv_combined = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        # Standard neural network layers
        self.relu = nn.ReLU()  # ReLU activation function
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        self.flatten = nn.Flatten()  # Flatten 2D features to 1D

        # Fully connected layers for final classification
        self.fc1__wo_pool = nn.Linear(32*22*22, 512)  # First FC layer (combined features)
        self.fc1__w_pool = nn.Linear(32*4*4, 512)     # Alternative FC layer (unused)
        self.fc2 = nn.Linear(512, 64)                 # Hidden layer
        self.fc3 = nn.Linear(64, num_classes)         # Final classification layer

        # Dropout layers for regularization
        self.drop1 = nn.Dropout(0.2)  # Light dropout after first FC
        self.drop2 = nn.Dropout(0.4)  # Heavier dropout before final layer

    def forward(self, x):
        """Forward pass through the parallel classical-quantum hybrid network.
        
        Processes the input simultaneously through classical and quantum branches,
        then fuses their features for improved classification performance.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            tuple: (predictions, conv1_quantum_embeddings, conv2_quantum_embeddings)
                - predictions: Class predictions of shape (batch_size, num_classes)
                - conv1_quantum_embeddings: First quantum layer outputs
                - conv2_quantum_embeddings: Second quantum layer outputs
        """
        # Classical CNN branch processing
        xc = self.relu(self.conv1(x))    # Shape: (batch_size, 16, 26, 26)
        xc = self.relu(self.conv2(xc))   # Shape: (batch_size, 32, 22, 22)

        # Quantum CNN branch processing (parallel to classical)
        conv1_out = self.conv1_PQK(x)               # First quantum layer
        x_q = self.relu(conv1_out)                  # Shape: (batch_size, 16, 26, 26)
        conv2_out = self.conv2__wo_pool_PQK(x_q)    # Second quantum layer
        x_q = self.relu(conv2_out)                  # Shape: (batch_size, 32, 22, 22)

        # Feature fusion: concatenate classical and quantum features
        x = torch.cat((x_q, xc), dim=1)  # Shape: (batch_size, 64, 22, 22)
        x = self.conv_combined(x)        # Shape: (batch_size, 32, 22, 22)

        # Final classification layers
        x = self.flatten(x)        # Shape: (batch_size, 32*22*22)
        x = self.fc1__wo_pool(x)   # Shape: (batch_size, 512)
        x = self.drop1(x)          # Apply dropout
        x = self.fc2(x)            # Shape: (batch_size, 64)
        x = self.drop2(x)          # Apply dropout
        x = self.fc3(x)            # Shape: (batch_size, num_classes)

        # Return predictions and quantum layer outputs (for analysis)
        return x, conv1_out.clone().detach(), conv2_out.clone().detach()


class LightweightCNN(nn.Module):
    """Lightweight classical CNN for comparison with quantum models.
    
    This model uses standard convolutional layers with similar architecture
    to the quantum models but with purely classical operations. It serves
    as a baseline for comparison with the quantum hybrid models.
    
    Args:
        num_classes (int): Number of output classes (default: 10 for MNIST digits)
    """
    def __init__(self, num_classes=10):
        super(LightweightCNN, self).__init__()
        
        # First convolutional layer: 1 input channel -> 16 output channels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        # Second convolutional layer: 16 input channels -> 32 output channels  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        
        # Standard neural network layers
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 11 * 11, 512)  # After conv layers: 32 channels, 11x11 feature maps
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Dropout for regularization
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.4)
    
    def forward(self, x):
        """Forward pass through the classical CNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            torch.Tensor: Class predictions of shape (batch_size, num_classes)
        """
        # First convolutional layer: (batch_size, 1, 28, 28) -> (batch_size, 16, 26, 26)
        x = self.relu(self.conv1(x))
        # Second convolutional layer: (batch_size, 16, 26, 26) -> (batch_size, 32, 22, 22)
        x = self.relu(self.conv2(x))
        # Max pooling: (batch_size, 32, 22, 22) -> (batch_size, 32, 11, 11)
        x = self.pool(x)
        
        # Flatten and pass through fully connected layers
        x = self.flatten(x)        # Shape: (batch_size, 32*11*11)
        x = self.fc1(x)            # Shape: (batch_size, 512)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)            # Shape: (batch_size, 64)
        x = self.drop2(x)
        x = self.relu(x)
        x = self.fc3(x)            # Shape: (batch_size, num_classes)
        
        return x
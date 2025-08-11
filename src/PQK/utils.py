# Import necessary libraries for quantum neural networks and data processing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import perceval as pcvl  # Quantum photonic simulation library
import merlin as ML      # Quantum machine learning library
import matplotlib.pyplot as plt


def complex_repeat_fix(tensor, *repeat_dims):
    """
    Workaround for PyTorch's repeat() not supporting complex tensors.
    Manually implements repeat functionality for complex tensors by splitting
    into real and imaginary parts, repeating separately, then recombining.
    
    Args:
        tensor (torch.Tensor): Complex tensor to repeat
        *repeat_dims: Dimensions along which to repeat the tensor
        
    Returns:
        torch.Tensor: Repeated complex tensor
    """
    if tensor.dtype in [torch.complex64, torch.complex128]:
        # Split into real and imaginary parts, repeat, then recombine
        real_part = tensor.real.repeat(*repeat_dims)
        imag_part = tensor.imag.repeat(*repeat_dims)
        return torch.complex(real_part, imag_part)
    else:
        # For non-complex tensors, use standard repeat
        return tensor.repeat(*repeat_dims)


# Monkey patch the torch.Tensor.repeat method for complex tensors
# Store original methods to avoid infinite recursion
original_repeat = torch.Tensor.repeat
original_scatter_add = torch.Tensor.scatter_add_

def patched_repeat(self, *repeat_dims):
    """Patched repeat method that handles complex tensors by delegating to complex_repeat_fix."""
    if self.dtype in [torch.complex64, torch.complex128]:
        return complex_repeat_fix(self, *repeat_dims)
    else:
        return original_repeat(self, *repeat_dims)

def patched_scatter_add(self, dim, index, src):
    """
    Patched scatter_add_ method that handles complex tensors.
    
    Complex tensors are handled by splitting into real and imaginary parts,
    applying scatter_add to each part separately, then recombining.
    
    Args:
        dim (int): Dimension along which to scatter
        index (torch.Tensor): Index tensor
        src (torch.Tensor): Source tensor to scatter
        
    Returns:
        torch.Tensor: Result of scatter_add operation
    """
    if self.dtype in [torch.complex64, torch.complex128] or src.dtype in [torch.complex64, torch.complex128]:
        # Convert to real dtype for scatter operations, then convert back
        if self.dtype in [torch.complex64, torch.complex128]:
            # Handle complex self tensor
            real_self = torch.stack([self.real, self.imag], dim=-1)
            if src.dtype in [torch.complex64, torch.complex128]:
                real_src = torch.stack([src.real, src.imag], dim=-1)
            else:
                # src is real, create imaginary part as zeros
                real_src = torch.stack([src, torch.zeros_like(src)], dim=-1)
            
            # Apply scatter_add to real and imaginary parts
            index_expanded = index.unsqueeze(-1).expand(*index.shape, 2)
            real_self.scatter_add_(dim, index_expanded, real_src)
            
            # Reconstruct complex tensor
            result = torch.complex(real_self[..., 0], real_self[..., 1])
            self.copy_(result)
            return self
        else:
            # self is real but src is complex - convert self to complex
            self_complex = torch.complex(self, torch.zeros_like(self))
            real_self = torch.stack([self_complex.real, self_complex.imag], dim=-1)
            real_src = torch.stack([src.real, src.imag], dim=-1)
            
            index_expanded = index.unsqueeze(-1).expand(*index.shape, 2)
            real_self.scatter_add_(dim, index_expanded, real_src)
            result = torch.complex(real_self[..., 0], real_self[..., 1])
            # Need to handle dtype conversion properly
            return result
    else:
        # For non-complex tensors, use original method
        return original_scatter_add(self, dim, index, src)

# Apply the monkey patches to PyTorch tensor methods
torch.Tensor.repeat = patched_repeat
torch.Tensor.scatter_add_ = patched_scatter_add


class QuantumEncodedDataset_type2(Dataset):
    """
    Custom PyTorch Dataset for quantum-encoded image data.
    
    This dataset handles loading and preprocessing of image data for quantum neural networks.
    It normalizes pixel values and optionally selects specific channels.
    
    Args:
        image_tensor (np.ndarray): Array of images 
        labels_tensor (np.ndarray): Array of corresponding labels
        num_samples (int): Number of samples in the dataset
        selected_channels (list, optional): Specific channels to use (None for all)
    """
    def __init__(self, image_tensor, labels_tensor, num_samples, selected_channels=None):
        self.image_tensor = image_tensor
        self.labels_tensor = labels_tensor
        self.num_samples = num_samples
        self.selected_channels = selected_channels

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (encoded_image_tensor, label_tensor)
                - encoded_image_tensor: Normalized image tensor
                - label_tensor: Corresponding label tensor
        """
        # Convert numpy arrays to PyTorch tensors and normalize pixel values to [0,1]
        encoded_image_tensor = torch.from_numpy(self.image_tensor[idx]) / 255.0
        label_tensor = torch.from_numpy(np.array(self.labels_tensor[idx]))

        # Optionally select specific channels if specified
        if self.selected_channels is not None:
            encoded_image_tensor = encoded_image_tensor[self.selected_channels, :, :]

        return encoded_image_tensor, label_tensor


def encoding_layer(type=2, kernel_size=2):
    """
    Create a quantum encoding circuit for processing classical data.
    
    This function generates different types of quantum encoding circuits based on
    the specified type. Type 2 uses delayed encoding which is more efficient
    for classical data encoding.
    
    Args:
        type (int): Type of encoding circuit (1 or 2, default: 2)
        kernel_size (int): Size of the kernel (default: 2)
        
    Returns:
        tuple: (circuit, modes)
            - circuit: Perceval quantum circuit for encoding
            - modes: Number of quantum modes used
    """
    if type == 1:
        # Standard encoding: one mode per pixel
        modes = kernel_size ** 2
    elif type == 2:  # delayed encoding
        # More efficient encoding: half the modes (rounded up for odd numbers)
        modes = kernel_size ** 2
        if (modes % 2) == 1:
            modes += 1  # Ensure even number of modes
        modes //= 2

    # Create quantum circuit with the calculated number of modes
    circuit = pcvl.Circuit(modes)

    if type == 1:
        # Type 1 encoding: direct phase shift encoding
        for i in range(kernel_size**2):
            circuit.add(i % modes, pcvl.PS(pcvl.P(f"px{i}")))  # Phase shift for each pixel
        
        # Add beam splitters for entanglement - even modes
        for i in range(0, modes, 2):
            try:
                circuit.add((i, (i + 1) % modes), pcvl.BS())  # Beam splitter between adjacent modes
            except Exception:
                pass
        
        # Add beam splitters for entanglement - odd modes
        for i in range(1, modes, 2):
            try:
                circuit.add((i, (i + 1) % modes), pcvl.BS())
            except Exception:
                pass

    elif type == 2:
        # Type 2 encoding: delayed encoding pattern
        # First pass: encode even-indexed pixels
        for i in range(kernel_size**2):
            if (i % 2) == 0:  # Only even-indexed pixels
                circuit.add(i % modes, pcvl.PS(pcvl.P(f"px{i}")))
        
        # Add beam splitters between even modes
        for i in range(0, modes, 2):
            try:
                circuit.add((i, (i + 1) % modes), pcvl.BS())
            except Exception:
                pass
        
        # Second pass: encode odd-indexed pixels
        for i in range(kernel_size**2):
            if (i % 2) == 1:  # Only odd-indexed pixels
                circuit.add(i % modes, pcvl.PS(pcvl.P(f"px{i}")))
        
        # Add beam splitters between odd modes
        for i in range(1, modes, 2):
            try:
                circuit.add((i, (i + 1) % modes), pcvl.BS())
            except Exception:
                pass

    return circuit, modes


def post_encoding__trainable(circuit, modes, start_idx, depth=2):
    """
    Add trainable quantum layers after the encoding layer.
    
    This function extends the quantum circuit with trainable parameters that can
    be optimized during training. The layers include permutations, phase shifts,
    and beam splitters.
    
    Args:
        circuit (pcvl.Circuit): Existing quantum circuit to extend
        modes (int): Number of quantum modes
        start_idx (int): Starting index for parameter naming
        depth (int): Number of trainable layers to add (default: 2)
    """
    # Add initial permutation layer
    perm = list(range(modes))
    circuit.add(0, pcvl.PERM(perm))
    
    # Add trainable layers with parametrized components
    for layer_idx in range(depth - 1):
        # Add trainable phase shifts for each mode
        for mode_idx in range(modes):
            circuit.add(mode_idx, pcvl.PS(pcvl.P(f"theta{start_idx}{layer_idx}{mode_idx}")))

        # Add beam splitters between even modes
        for i in range(0, modes, 2):
            try:
                circuit.add((i, (i + 1) % modes), pcvl.BS())
            except Exception:
                pass
        
        # Add beam splitters between odd modes
        for i in range(1, modes, 2):
            try:
                circuit.add((i, (i + 1) % modes), pcvl.BS())
            except Exception:
                pass
        
        # Add permutation layer
        perm = list(range(modes))
        circuit.add(0, pcvl.PERM(perm))
    
    # Final beam splitter layers without additional phase shifts
    for i in range(0, modes, 2):
        try:
            circuit.add((i, (i + 1) % modes), pcvl.BS())
        except Exception:
            pass
    for i in range(1, modes, 2):
        try:
            circuit.add((i, (i + 1) % modes), pcvl.BS())
        except Exception:
            pass


def generate_quantum_embedding__trainable(modes=9, depth=2, re_encoding=1, layer_global_idx=1, session=None):
    """
    Generate a complete trainable quantum embedding circuit.
    
    This function creates a quantum circuit that combines encoding layers with
    trainable post-processing layers. Multiple re-encoding passes can be applied
    for more complex transformations.
    
    Args:
        modes (int): Number of quantum modes (default: 9)
        depth (int): Depth of trainable layers (default: 2)
        re_encoding (int): Number of re-encoding passes (default: 1)
        layer_global_idx (int): Global layer identifier (default: 1)
        session: Session parameter (unused, for compatibility)
        
    Returns:
        pcvl.Circuit: Complete quantum embedding circuit
    """
    # Generate multiple encoding passes if requested
    for re_encoding_idx in range(re_encoding):
        # Create encoding circuit with 3x3 kernel (hardcoded for this implementation)
        encode_circuit, modes_ret = encoding_layer(kernel_size=3)
        # Add trainable post-processing layers
        post_encoding__trainable(
            encode_circuit, 
            modes_ret, 
            start_idx=int(str(layer_global_idx) + str(re_encoding_idx)), 
            depth=depth
        )
    
    return encode_circuit


class QuantumConv2D__Trainable_Type2(nn.Module):
    """
    Trainable quantum convolutional layer using Type 2 encoding.
    
    This layer implements quantum convolution by processing image patches through
    a parameterized quantum circuit. It uses the Merlin library to interface
    between classical neural networks and quantum circuits.
    
    Args:
        img_height (int): Input image height
        img_width (int): Input image width  
        kernel_size (tuple): Size of the convolutional kernel (height, width)
        stride (int): Convolution stride (default: 1)
        in_channels (int): Number of input channels (default: 1)
        output_size_merlin (int): Number of output feature maps (default: 6)
        modes_merlin (int): Number of quantum modes (default: 9)
        depth_merlin (int): Depth of quantum circuit (default: 2)
        layer_global_idx (int): Global layer identifier (default: 1)
        device (str): Device to run on ('cpu', 'cuda', or 'mps')
    """
    def __init__(self, img_height, img_width, kernel_size=(3, 3), stride=1,
                 in_channels=1, output_size_merlin=6, modes_merlin=9, depth_merlin=2, 
                 layer_global_idx=1, device='cpu'):
        super(QuantumConv2D__Trainable_Type2, self).__init__()
        
        # Store layer parameters
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels

        # Generate the quantum circuit for this layer
        self.perceval_circuit = generate_quantum_embedding__trainable(
            modes=modes_merlin,
            depth=depth_merlin,
            re_encoding=2,  # Use 2 re-encoding passes for better feature extraction
            layer_global_idx=layer_global_idx
        )

        # Create Merlin quantum layer that interfaces with PyTorch
        self.merlin_quantum_layer = ML.QuantumLayer(
            input_size=kernel_size[0]*kernel_size[1],  # Flattened kernel size
            output_size=output_size_merlin,            # Number of output features
            circuit=self.perceval_circuit,             # Quantum circuit
            trainable_parameters=["theta"],            # Parameters to optimize
            input_parameters=["px"],                   # Input encoding parameters
            input_state=[1, 0] * (modes_merlin // 2) + [0] * (modes_merlin % 2),  # Initial quantum state
            output_mapping_strategy=ML.OutputMappingStrategy.LINEAR,  # Linear output mapping
            device=device
        )

        # Calculate output dimensions after convolution
        self.output_height = (img_height - kernel_size[0]) // stride + 1
        self.output_width = (img_width - kernel_size[1]) // stride + 1

    def forward(self, x):
        """
        Forward pass through the quantum convolutional layer.
        
        This method applies quantum convolution by:
        1. Extracting image patches using unfold operation
        2. Processing each patch through the quantum circuit
        3. Reshaping the output to form feature maps
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output feature maps of shape (batch_size, output_size_merlin, output_height, output_width)
        """
        batch_size, _, img_height, img_width = x.shape

        # Extract patches from the input image using unfold operation
        # unfold extracts all possible kernel_size patches with given stride
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        num_patches_per_image = patches.shape[2]
        
        # Reshape patches for processing: (batch_size, num_patches, kernel_height*kernel_width*channels)
        patches_flat = patches.permute(0, 2, 1).reshape(-1, self.kernel_size[0] * self.kernel_size[1] * self.in_channels)

        # Process all patches through the quantum layer
        quantum_features_flat = self.merlin_quantum_layer(patches_flat)

        # Reshape quantum features back to spatial format
        # First reshape to (batch_size, num_patches, output_size)
        quantum_feature_map = quantum_features_flat.view(
            batch_size,
            num_patches_per_image,
            self.merlin_quantum_layer.output_size
        ).permute(0, 2, 1)  # Permute to (batch_size, output_size, num_patches)

        # Finally reshape to proper 2D feature maps
        quantum_feature_map = quantum_feature_map.view(
            batch_size,
            self.merlin_quantum_layer.output_size,
            self.output_height,
            self.output_width
        )

        return quantum_feature_map


def plot_training_metrics(accuracies, losses, title_prefix="Training"):
    """
    Plot training accuracy and loss curves for model evaluation.
    
    Creates a side-by-side visualization of training progress with accuracy
    and loss curves over epochs.
    
    Args:
        accuracies (list): List of accuracy values per epoch
        losses (list): List of loss values per epoch  
        title_prefix (str): Prefix for plot titles (default: "Training")
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(accuracies) + 1)
    
    # Plot accuracy curve
    ax1.plot(epochs, accuracies, 'b-', label='Accuracy')
    ax1.set_title(f'{title_prefix} Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True)
    
    # Plot loss curve
    ax2.plot(epochs, losses, 'r-', label='Loss')
    ax2.set_title(f'{title_prefix} Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def get_device():
    """
    Get the best available computing device for PyTorch operations.
    
    Currently forces CPU usage due to MPS compatibility issues with complex
    tensor operations in the Merlin quantum library. This should be updated
    when MPS/CUDA support is added to the quantum computing libraries.
    
    Returns:
        torch.device: Device object ('cpu', 'cuda', or 'mps')
    """
    # Force CPU for now due to MPS compatibility issues with complex tensor operations
    # TODO: Re-enable MPS/CUDA once Merlin library supports these backends properly
    device = "cpu"  # Commented: "cuda" if torch.cuda.is_available() else "cpu"
    # Commented: device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else device
    return torch.device(device)



def create_dataloaders(train_images, test_images, train_labels, test_labels, batch_size=32, num_workers=2):
    """
    Create PyTorch DataLoaders for efficient batch processing during training and testing.
    
    Args:
        train_images (np.ndarray): Training images
        test_images (np.ndarray): Test images
        train_labels (np.ndarray): Training labels  
        test_labels (np.ndarray): Test labels
        batch_size (int): Batch size for training (default: 32)
        num_workers (int): Number of parallel data loading workers (default: 2)
        
    Returns:
        tuple: (train_dataloader, test_dataloader)
            DataLoader objects ready for training and evaluation
    """
    # Create custom datasets using our quantum-aware dataset class
    train_dataset = QuantumEncodedDataset_type2(
        image_tensor=train_images,
        labels_tensor=train_labels,
        num_samples=len(train_labels),
        selected_channels=None  # Use all channels
    )

    test_dataset = QuantumEncodedDataset_type2(
        image_tensor=test_images,
        labels_tensor=test_labels,
        num_samples=len(test_labels),
        selected_channels=None  # Use all channels
    )

    # Create DataLoaders with specified batch size and parallel loading
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,      # Shuffle training data for better learning
        num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,     # Don't shuffle test data
        num_workers=num_workers
    )
    
    return train_dataloader, test_dataloader


def evaluate_model(model, test_dataloader, criterion, device):
    """
    Evaluate a trained model on test data and return performance metrics.
    
    This function sets the model to evaluation mode, disables gradient computation
    for efficiency, and computes test loss and accuracy over the entire test set.
    
    Args:
        model (nn.Module): Trained PyTorch model to evaluate
        test_dataloader (DataLoader): DataLoader containing test data
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss)
        device (torch.device): Device to run evaluation on
        
    Returns:
        tuple: (test_loss, test_accuracy)
            - test_loss: Average loss over test set
            - test_accuracy: Accuracy percentage over test set
    """
    # Set model to evaluation mode (disables dropout, batch norm training mode, etc.)
    model.eval()
    
    # Initialize metrics
    test_loss = 0.0
    correct_test_predictions = 0
    total_test_predictions = 0

    # Disable gradient computation for faster evaluation
    with torch.no_grad():
        for images, labels in test_dataloader:
            # Move data to the appropriate device
            images, labels = images.to(device), labels.to(device)
            
            # Handle models that return embeddings vs. those that don't
            if len(model(images)) == 3:  # Model returns (outputs, embedding1, embedding2)
                outputs, _, _ = model(images)
            else:
                outputs = model(images)
                
            # Calculate loss
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)  # Accumulate weighted loss

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class indices
            total_test_predictions += labels.size(0)
            correct_test_predictions += (predicted == labels).sum().item()

    # Calculate average loss and accuracy
    test_loss /= len(test_dataloader.dataset)
    test_accuracy = 100 * correct_test_predictions / total_test_predictions

    return test_loss, test_accuracy
from torch import nn, optim
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
import random
from PIL import ImageFilter, Image
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import perceval as pcvl
from merlin.datasets import mnist_digits
import datetime


def set_seed(seed=42):
    """
    Set the random seed for reproducibility across different libraries.

    Args:
        seed (int): Seed value to use. Default is 42.
    """
    # Set Python's random seed
    random.seed(seed)

    # Set NumPy's random seed
    np.random.seed(seed)

    # Set PyTorch's random seeds for both CPU and CUDA
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Additional settings for complete reproducibility
    # Note: This can affect performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For some PyTorch operations using Intel MKL
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seed set to {seed}")

#####################
### DATA HANDLING ###
#####################

def crop_square(args, image, position='upper_left'):
    """
    Crop a square from a numpy array based on the specified position.

    Args:
        image: Numpy array of shape (H, W, C) or (H, W)
        position: Either 'upper_left' or 'bottom_right'

    Returns:
        Cropped square numpy array
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape

    # Determine size of the square (the smaller dimension)
    size = args.size

    if position == 'upper_left':
        if args.cnn:
            return image[:size, :size].copy()
        else:
            return image[:size, :size].reshape((size * size)).copy()
    elif position == 'bottom_right':
        if args.cnn:
            return image[height - size:, width - size:].copy()
        else:
            return image[height - size:, width - size:].reshape((size * size)).copy()
    else:
        raise ValueError(f"Unsupported position: {position}")


def resize_and_flatten(tensor, target_size=(20, 20)):
    """
    Resize a tensor from [batch_size, H, W] to [batch_size, target_size[0], target_size[1]]
    and then flatten to [batch_size, target_size[0]*target_size[1]]
    """
    batch_size = tensor.shape[0]

    # Add channel dimension, resize, then remove channel dimension
    resized = F.interpolate(
        tensor.unsqueeze(1),
        size=target_size,
        mode='bilinear',
        align_corners=False
    ).squeeze(1)

    # Flatten the spatial dimensions
    flattened = resized.reshape(batch_size, -1)

    return flattened


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TransformCrop:
    def __init__(self, args, blur_prob=0.1):
        self.blur_prob = blur_prob
        self.gaussian_blur = GaussianBlur([0.1, 2.0])
        self.transform = lambda x: crop_square(args, x, 'upper_left')
        self.transform_prime = lambda x: crop_square(args, x, 'bottom_right')

    def __call__(self, x):
        # Convert numpy array to PIL Image for blur transformation
        if isinstance(x, np.ndarray):
            # Normalize to 0-255 range for PIL
            if x.max() <= 1.0:
                x_pil = Image.fromarray((x * 255).astype(np.uint8))
            else:
                x_pil = Image.fromarray(x.astype(np.uint8))
        else:
            x_pil = x

        # Apply Gaussian blur with low probability
        x_pil_1 = np.array(self.gaussian_blur(x_pil)) / 255.0 if random.random() < self.blur_prob else np.array(
            x_pil) / 255.0
        x_pil_2 = np.array(self.gaussian_blur(x_pil)) / 255.0 if random.random() < self.blur_prob else np.array(
            x_pil) / 255.0

        y1 = self.transform(x_pil_1).astype(np.float32)
        y2 = self.transform_prime(x_pil_2).astype(np.float32)
        return y1, y2


class TransformedDataset(Dataset):
    def __init__(self, data, transform=None, labels=None):
        """
        Args:
            data: List of numpy arrays
            transform: Transform to apply to each array
            labels: Optional labels for supervised learning
        """
        self.data = data
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]

        if self.transform:
            transformed_image = self.transform(image)
        else:
            transformed_image = image

        if self.labels is not None:
            return transformed_image, self.labels[idx]

        return transformed_image


def load_training_data(args):
    X_train, y_train, metadata = mnist_digits.get_data_train_percevalquest()
    X_val, y_val, metadata = mnist_digits.get_data_test_percevalquest()

    transfo = TransformCrop(args, blur_prob=0.0)

    # we do not need the labels at that point
    train_data = TransformedDataset(X_train, transform=transfo)
    val_data = TransformedDataset(X_val, transform=transfo)

    return train_data, val_data


def load_finetuning_data(args):
    X_train, y_train, metadata = mnist_digits.get_data_train_percevalquest()
    X_val, y_val, metadata = mnist_digits.get_data_test_percevalquest()

    # Transform to 20x20 shape (400 total) for fine-tuning
    def to_tensor_transform(x):
        # Resize from 28x28 to 20x20, then flatten to 400 elements
        if len(x.shape) == 1:
            # If already flattened, reshape to 28x28 first
            x = x.reshape(28, 28)
        # Resize to 20x20 using PIL
        x_pil = Image.fromarray((x * 255).astype(np.uint8) if x.max() <= 1.0 else x.astype(np.uint8))
        x_resized = x_pil.resize((args.size, args.size), Image.LANCZOS)
        x_array = np.array(x_resized).astype(np.float32) / 255.0
        if args.cnn:
            return torch.tensor(x_array, dtype=torch.float32).unsqueeze(0)
        else:
            return torch.tensor(x_array.flatten(), dtype=torch.float32)

    train_data = TransformedDataset(X_train, transform=to_tensor_transform, labels=y_train)
    val_data = TransformedDataset(X_val, transform=to_tensor_transform, labels=y_val)

    return train_data, val_data

###############
### ENCODER ###
###############

### classical encoder ###
# MLP encoder
class CustomMLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        dims = args.enc_dim
        if dims == 0 or dims == [0]:
            mlp = [nn.Linear(args.size**2,args.backbone_dim)]
        else:
            mlp = []
            dims = [args.size**2] + dims
            for k in range(len(dims)-1):
                mlp.append(nn.Linear(dims[k],dims[k+1]))
                mlp.append(nn.ReLU())
            mlp.append(nn.Linear(dims[-1],args.backbone_dim))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)

# CNN encoder
class CustomCNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        channels = args.enc_dim

        if channels == 0 or channels == [0] or len(channels) == 0:
            # Simple case: single conv layer + global pooling + classifier
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(32, args.backbone_dim)

        else:
            # Custom architecture case
            channels = [1] + list(channels)  # Start with 1 input channel

            # Build conv layers
            self.conv_layers = nn.ModuleList()
            self.pool_layers = nn.ModuleList()

            for k in range(len(channels)-1):
                self.conv_layers.append(nn.Conv2d(channels[k], channels[k+1], kernel_size=3, padding=1))
                # Add pooling for all but the last layer (matching SimpleCNN pattern)
                if k < len(channels) - 2:  # Don't pool after the last conv layer
                    self.pool_layers.append(nn.MaxPool2d(2, 2))
                else:
                    self.pool_layers.append(None)

            # Calculate final feature map size
            # Each pooling operation divides by 2
            num_pools = len([p for p in self.pool_layers if p is not None])
            final_size = args.size // (2 ** num_pools)

            self.fc1 = nn.Linear(final_size * final_size * channels[-1], args.backbone_dim)

    def forward(self, x):
        if hasattr(self, 'adaptive_pool'):
            # Simple case - single conv + global pooling
            x = F.relu(self.conv1(x))
            x = self.adaptive_pool(x)
            x = x.view(x.size(0), -1)  # Flatten
            out = F.relu(self.fc1(x))


        else:
            # Custom architecture case
            for i, (conv, pool) in enumerate(zip(self.conv_layers, self.pool_layers)):
                x = F.relu(conv(x))
                if pool is not None:
                    x = pool(x)

            # Flatten
            x = x.view(x.size(0), -1)

            # Fully connected layer with dropout
            out = F.relu(self.fc1(x))

        return out

### quantum circuit ###
def create_quantum_circuit(modes = 10, feature_size = 10):
    # first trainable circuit
    """pre_circuit = pcvl.GenericInterferometer(modes,
                               lambda i: pcvl.BS() ,
                               shape=pcvl.InterferometerShape.TRIANGLE)"""
    pre_circuit = pcvl.GenericInterferometer(modes, lambda i: (pcvl.BS()  # theta=pcvl.P(f"bs_1_{i}")
                                                                    .add(0, pcvl.PS(pcvl.P(f"phase_train_1_{i}")))
                                                                    .add(0, pcvl.BS())  # theta=pcvl.P(f"bs_1_{i}")
                                                                    .add(0, pcvl.PS(pcvl.P(f"phase_train_2_{i}")))
                                                                    )
                                                  )
    # data encoding in phase shifters (sandwhich)
    var = pcvl.Circuit(modes)
    for k in range(0, feature_size):
        var.add(k % modes, pcvl.PS(pcvl.P(f"feature-{k}")))

    # second trainable circuit
    post_circuit = pcvl.GenericInterferometer(modes, lambda i: (pcvl.BS()  # theta=pcvl.P(f"bs_1_{i}")
                                                                .add(0, pcvl.PS(pcvl.P(f"phase_train_3_{i}")))
                                                                .add(0, pcvl.BS())  # theta=pcvl.P(f"bs_1_{i}")
                                                                .add(0, pcvl.PS(pcvl.P(f"phase_train_4_{i}")))
                                                                )
                                              )
    """post_circuit = pcvl.GenericInterferometer(modes,
                               lambda i: pcvl.BS(),
                               shape=pcvl.InterferometerShape.TRIANGLE)"""
    circuit = pcvl.Circuit(modes)

    circuit.add(0, pre_circuit, merge=True)
    circuit.add(0, var, merge=True)
    circuit.add(0, post_circuit, merge=True)

    return circuit



#################
### CRITERION ###
#################

### InfoNCE Loss ###
class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, x1, x2):
        """
        Args:
            features: Tensor of shape (2 * batch_size, feature_dim).
                      The first half are augmented views of instances in the batch.
                      The second half are corresponding positive pairs.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        features = torch.cat([x1, x2], dim=0)
        #print(f"\n -- in Loss, dimension x1: {x1.shape} and dimension features: {features.shape}")
        batch_size = features.shape[0] // 2
        # create pseudo labels
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(features.device)
        # sim(z_i,z_j)/tau
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # mask to remove self-comparisons
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        #print(f"\n Similarity matrix:\n{similarity_matrix}")
        # exp(sim(z_i,z_j)/tau)
        sim_exp = torch.exp(similarity_matrix)
        # exp(sim(z_i,z_j)/tau) - same indices
        sim_exp_sum = sim_exp.sum(dim = 1, keepdim = True) - torch.exp(similarity_matrix.diagonal().view(-1,1))
        # - log( # exp(sim(z_i,z_j)/tau) / sum )
        log_prob = similarity_matrix - torch.log(sim_exp_sum+1e-8)

        pos_indices = torch.arange(batch_size).to(features.device)
        pos_pairs = torch.cat([pos_indices+batch_size,pos_indices]).to(features.device)
        # Apply softmax to get probabilities
        loss = -log_prob[torch.arange(2*batch_size),pos_pairs].mean()

        #print(f"Loss: {loss}")
        return loss.mean()

# KL divergence
class KLDivergenceLoss(nn.Module):
    """
    KL Divergence Loss as nn.Module for use as loss function
    """
    def __init__(self, eps=1e-8, reduction='mean'):
        super(KLDivergenceLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, p, q):
        """
        Compute KL divergence between two distributions
        Args:
            p: tensor of shape [batch_size, dim] - first distribution
            q: tensor of shape [batch_size, dim] - second distribution
        Returns:
            KL(p||q) loss
        """
        kl_div = torch.sum(p * (torch.log(p + self.eps) - torch.log(q + self.eps)), dim=-1)
        
        if self.reduction == 'mean':
            return torch.mean(kl_div)
        elif self.reduction == 'sum':
            return torch.sum(kl_div)
        else:
            return kl_div

# Jensen-Shannon divergence
class JSDivergenceLoss(nn.Module):
    """
    Jensen-Shannon Divergence Loss as nn.Module for use as loss function
    """
    def __init__(self, eps=1e-8, reduction='mean'):
        super(JSDivergenceLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, p, q):
        """
        Compute Jensen-Shannon divergence between two distributions
        Args:
            p: tensor of shape [batch_size, dim] - first distribution
            q: tensor of shape [batch_size, dim] - second distribution
        Returns:
            JS(p||q) loss
        """
        m = 0.5 * (p + q)
        kl_pm = torch.sum(p * (torch.log(p + self.eps) - torch.log(m + self.eps)), dim=-1)
        kl_qm = torch.sum(q * (torch.log(q + self.eps) - torch.log(m + self.eps)), dim=-1)
        js_div = 0.5 * kl_pm + 0.5 * kl_qm
        
        if self.reduction == 'mean':
            return torch.mean(js_div)
        elif self.reduction == 'sum':
            return torch.sum(js_div)
        else:
            return js_div


# Total Variation Distance
class TotalVariationLoss(nn.Module):
    """
    Total Variation Distance Loss as nn.Module for use as loss function
    """
    def __init__(self, reduction='mean'):
        super(TotalVariationLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, p, q):
        """
        Compute Total Variation Distance between two distributions
        Args:
            p: tensor of shape [batch_size, dim] - first distribution
            q: tensor of shape [batch_size, dim] - second distribution
        Returns:
            TVD loss
        """
        # Check shapes
        if len(p.shape) != 2 or len(q.shape) != 2:
            raise ValueError("Input tensors must be 2D with shape (batch_size, dim)")

        batch_size_p, dim_p = p.shape
        batch_size_q, dim_q = q.shape

        if batch_size_p != batch_size_q:
            raise ValueError(f"Batch sizes must match: p has {batch_size_p}, q has {batch_size_q}")

        if dim_p != dim_q:
            raise ValueError(f"Distributions must have same dimensions: p has {dim_p}, q has {dim_q}")

        # Calculate total variation distance for each pair in the batch
        tvd = 0.5 * torch.sum(torch.abs(p - q), dim=1)
        
        if self.reduction == 'mean':
            return torch.mean(tvd)
        elif self.reduction == 'sum':
            return torch.sum(tvd)
        else:
            return tvd


##########################################
### FUNCTIONS TO PLOT AND SAVE RESULTS ###
##########################################

### training and evaluation metrics ###

def plot_training_loss(training_losses, args, results_dir):
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, args.epochs + 1), training_losses, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'SSL Training Loss ({"Quantum" if args.quantum else "Classical"} Network)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'ssl_training_loss_{"quantum" if args.quantum else "classical"}.png'),
                dpi=300, bbox_inches='tight')
    if not args.no_plots:
        plt.show()
    plt.close()


def plot_evaluation_metrics(train_losses, val_losses, train_accs, val_accs, args, results_dir):
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot losses
    epochs = range(1, args.ft_epochs + 1)
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Fine-tuning Losses ({"Quantum" if args.quantum else "Classical"} Network)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', linewidth=2, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Fine-tuning Accuracies ({"Quantum" if args.quantum else "Classical"} Network)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'evaluation_metrics_{"quantum" if args.quantum else "classical"}.png'),
                dpi=300, bbox_inches='tight')
    if not args.no_plots:
        plt.show()
    plt.close()


def save_results_to_json(args, ssl_training_losses, ft_train_losses, ft_val_losses, ft_train_accs, ft_val_accs,
                         results_dir, random_results=None):
    # Determine filename based on quantum mode
    filename = os.path.join(results_dir, f'{"quantum" if args.quantum else "classical"}_results.json')

    # Create experiment entry
    experiment = {
        'timestamp': datetime.datetime.now().isoformat(),
        'arguments': vars(args),
        'ssl_training_losses': ssl_training_losses,
        'fine_tuning': {
            'train_losses': ft_train_losses,
            'val_losses': ft_val_losses,
            'train_accuracies': ft_train_accs,
            'val_accuracies': ft_val_accs,
            'final_val_accuracy': ft_val_accs[-1] if ft_val_accs else 0.0,
            'best_val_accuracy': max(ft_val_accs) if ft_val_accs else 0.0
        }
    }

    # Add random model results if provided
    if random_results:
        experiment['random_baseline'] = {
            'train_losses': random_results['train_losses'],
            'val_losses': random_results['val_losses'],
            'train_accuracies': random_results['train_accs'],
            'val_accuracies': random_results['val_accs'],
            'final_val_accuracy': random_results['val_accs'][-1] if random_results['val_accs'] else 0.0,
            'best_val_accuracy': max(random_results['val_accs']) if random_results['val_accs'] else 0.0
        }

    # Load existing results or create new list
    try:
        with open(filename, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []

    # Append new experiment
    results.append(experiment)

    # Save updated results
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {filename}")
    print(f"Final validation accuracy: {experiment['fine_tuning']['final_val_accuracy']:.4f}")
    print(f"Best validation accuracy: {experiment['fine_tuning']['best_val_accuracy']:.4f}")


### confusion matrix ###

# the function below is made 99% by Claude and 1% by me :)
def compute_confusion_matrix(model, val_loader, class_names=None, device='cuda'):
    """
    Compute confusion matrix for a single trained model using a validation data loader.

    Args:
        model: PyTorch model
        val_loader: PyTorch DataLoader containing validation data
        class_names: List of class names (optional)
        device: Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        confusion_matrix: numpy array of the confusion matrix
        accuracy: float accuracy score
    """
    # Set model to evaluation mode
    model.eval()

    # Move model to the appropriate device
    model = model.to(device)

    # Initialize lists to store predictions and ground truth
    all_preds = []
    all_targets = []

    # Disable gradient computation for inference
    with torch.no_grad():
        for inputs, targets in val_loader:
            # Handle input format properly - don't resize if already in correct format
            if len(inputs.shape) == 4 and inputs.shape[1] == 1:  # CNN format (batch, 1, H, W)
                inputs = inputs.to(device)
            elif len(inputs.shape) == 3:  # (batch, H, W) - need to resize and flatten
                inputs = resize_and_flatten(inputs).to(device)
            else:  # Already flattened (batch, features)
                inputs = inputs.to(device)
            targets = targets.to(device)

            # Get predictions from model
            outputs = model(inputs)

            # Convert outputs to class predictions
            _, preds = torch.max(outputs, 1)

            # Append batch predictions and targets to lists
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Calculate accuracy
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)

    return cm, accuracy


### TSNE ###

def extract_features(model, dataloader, device='cuda'):
    model.eval()  # Set the model to evaluation mode
    features = []
    labels = []

    with torch.no_grad():
        for data, label in dataloader:
            # Handle input format properly - don't resize if already in correct format
            if len(data.shape) == 4 and data.shape[1] == 1:  # CNN format (batch, 1, H, W)
                data = data.to(device)
            elif len(data.shape) == 3:  # (batch, H, W) - need to resize and flatten
                data = resize_and_flatten(data).to(device)
            else:  # Already flattened (batch, features)
                data = data.to(device)
            output = model(data)
            features.append(output.cpu())  # Move to CPU for compatibility
            labels.extend(label.cpu().numpy())

    features = torch.cat(features, dim=0).numpy()
    labels = torch.tensor(labels).numpy()
    return features, labels



def compute_tsne(model, val_loader, device='cpu'):
    """
    Compute t-SNE for a single trained model using a validation data loader.

    Args:
        model: PyTorch model
        val_loader: PyTorch DataLoader containing validation data
        device: Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        features_2d: numpy array of t-SNE reduced features (n_samples, 2)
        labels: numpy array of true labels
    """
    # Set model to evaluation mode
    model.eval()

    # Move model to the appropriate device
    model = model.to(device)

    # Extract features and labels
    features, labels = extract_features(model, val_loader, device=device)
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    return features_2d, labels


def plot_tsne(features_2d, labels, model_name, results_dir, args):
    """
    Plot and save t-SNE visualization for a single model.
    
    Args:
        features_2d: numpy array of t-SNE reduced features (n_samples, 2)
        labels: numpy array of true labels
        model_name: string name for the model (e.g., 'fine_tuned', 'random')
        results_dir: directory to save the plot
        args: arguments containing model configuration
    """
    plt.figure(figsize=(10, 8))
    
    # Plot each class with different colors
    num_classes = len(np.unique(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    for class_idx in range(num_classes):
        mask = labels == class_idx
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=[colors[class_idx]], label=f'Digit {class_idx}', alpha=0.6, s=20)
    
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f't-SNE Visualization - {model_name.replace("_", " ").title()} Model')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    filename = f'tsne_{model_name}_{"quantum" if args.quantum else "classical"}.png'
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    if not args.no_plots:
        plt.show()
    plt.close()
    
    print(f"t-SNE plot saved to: {filepath}")


def plot_confusion_matrix(cm, accuracy, model_name, results_dir, args, class_names=None):
    """
    Plot and save confusion matrix for a single model.
    
    Args:
        cm: confusion matrix (numpy array)
        accuracy: accuracy score (float)
        model_name: string name for the model (e.g., 'fine_tuned', 'random')
        results_dir: directory to save the plot
        args: arguments containing model configuration
        class_names: list of class names (optional)
    """
    plt.figure(figsize=(8, 6))
    
    # Create confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    
    plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()} Model\nAccuracy: {accuracy:.4f}')
    plt.tight_layout()
    
    # Save the plot
    filename = f'confusion_matrix_{model_name}_{"quantum" if args.quantum else "classical"}.png'
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    if not args.no_plots:
        plt.show()
    plt.close()
    
    print(f"Confusion matrix saved to: {filepath}")


if __name__ == "__main__":
    import argparse
    
    # Create parser with default arguments from SSL_qLoss.py
    parser = argparse.ArgumentParser(description='SSL with Quantum Loss')
    parser.add_argument('-cl', '--classes', type=int, default=10, help='Number of classes')
    parser.add_argument('-s', '--size', type=int, default=20, help='size x size of the image')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('-bn', '--batch_norm', action='store_true', default=False,
                        help='Set if we use BatchNorm after compression of the encoder')
    parser.add_argument('-bck-d', '--backbone-dim', type=int, default=64, help='Dimension of the backbone output')
    parser.add_argument('--cnn', action='store_true', default=False, help='backbone is CNN if True, MLP otherwise')
    parser.add_argument('--enc-dim', type=int, default=[8, 8], nargs='+',
                            help='Dimensions of the encoder')
    parser.add_argument('-hd', '--hidden_dim', type=int, default=20, help='Hidden dimension of the projector')
    parser.add_argument('-ld', '--loss_dim', type=int, default=20, help='Dimension of the loss space')
    parser.add_argument('-tau', '--temperature', type=float, default=0.07, help='Temperature of the InfoNCELoss')
    parser.add_argument('-w', '--width', type=int, default=8, help='Dimension of the features encoded in the QNN')
    parser.add_argument('-quant', '--quantum', action='store_true', default=False, help='Set if we use Quantum SSL')
    parser.add_argument('-m', '--modes', type=int, default=10, help='Number of modes')
    parser.add_argument('--no_bunching', action='store_true',
                            help='NoBunching mode for the quantumlayer')
    parser.add_argument('-d', '--datadir', type=str, default='./data', help='Data directory')
    parser.add_argument('--no-plots', action='store_true', default=False, help='Disable plot display')
    
    # Parse arguments with defaults
    args = parser.parse_args()
    
    # Load training data with args
    train_data, val_data = load_training_data(args)

    # Plot some samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Training Data Samples - Cropped Views', fontsize=16)

    for i in range(5):
        # Get a sample from training data
        view1, view2 = train_data[i]

        # Reshape back to size x size for visualization
        view1_img = view1.reshape(args.size, args.size)
        view2_img = view2.reshape(args.size, args.size)

        # Plot upper left crop
        axes[0, i].imshow(view1_img, cmap='gray')
        axes[0, i].set_title(f'Sample {i + 1}\nUpper Left')
        axes[0, i].axis('off')

        # Plot bottom right crop
        axes[1, i].imshow(view2_img, cmap='gray')
        axes[1, i].set_title(f'Sample {i + 1}\nBottom Right')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

    # Load fine-tuning data with args
    finetune_train_data, finetune_val_data = load_finetuning_data(args)

    # Plot some fine-tuning samples
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle('Fine-tuning Data Samples - Tensor Format', fontsize=16)

    for i in range(5):
        # Get a sample from fine-tuning data
        tensor_sample, label = finetune_train_data[i]

        # Reshape tensor back to original image dimensions for visualization
        if args.cnn:
            # For CNN, tensor_sample has shape (1, size, size)
            sample_img = tensor_sample.squeeze(0)
        else:
            # For MLP, tensor_sample is flattened
            sample_img = tensor_sample.reshape(args.size, args.size)

        # Plot the sample
        axes[i].imshow(sample_img, cmap='gray')
        axes[i].set_title(f'Sample {i + 1}\nLabel: {label}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
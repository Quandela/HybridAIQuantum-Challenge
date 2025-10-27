import perceval as pcvl
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import random
import numpy as np
import os
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import json
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from merlin.datasets import mnist_digits


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

##########################
### load MNIST dataset ###
##########################

def crop_middle(image, size = 20):
    """
    Crop the middle from a numpy array based on the specified position.

    Args:
        image: Numpy array of shape (H, W, C) or (H, W)
        size: total size of the cropped image

    Returns:
        Cropped square numpy array
    """
    if len(image.shape) == 3:
        _, height, width = image.shape
    else:
        height, width = image.shape

    # Determine size of the square (the smaller dimension)
    mid_x = int(width * 0.5)
    mid_y = int(height * 0.5)
    half_size = int(size / 2)

    return image[mid_x-half_size:mid_x+half_size, mid_y-half_size:mid_y+half_size].copy()

class TransformCenter:
    def __init__(self, size = 20):
        self.transform = lambda x: crop_middle(x, size = size)
        self.tensor_transform = ToTensor()
        self.size = size

    def __call__(self, x):
        y1 = self.tensor_transform(self.transform(x)).view(self.size * self.size)
        return y1


# load the correct train, val dataset for the challenge, from the csv files
class MNIST_partial(Dataset):
    def __init__(self, data='./data', transform=None, split='train'):
        """
        Args:
            data: path to dataset folder which contains train.csv and val.csv
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g., data augmentation or normalization)
            split: 'train' or 'val' to determine which set to download
        """
        self.data_dir = data
        self.transform = transform
        self.data = []

        if split == 'train':
            filename = os.path.join(self.data_dir, 'train.csv')
        elif split == 'val':
            filename = os.path.join(self.data_dir, 'val.csv')
        else:
            raise AttributeError("split!='train' and split!='val': split must be train or val")

        self.df = pd.read_csv(filename)

    def __len__(self):
        l = len(self.df['image'])
        return l

    def __getitem__(self, idx):
        img = self.df['image'].iloc[idx]
        label = self.df['label'].iloc[idx]
        # string to list
        img_list = re.split(r',', img)
        # remove '[' and ']'
        img_list[0] = img_list[0][1:]
        img_list[-1] = img_list[-1][:-1]
        # convert to float
        img_float = [float(el) for el in img_list]
        # convert to image
        img_square = torch.unflatten(torch.tensor(img_float), 0, (1, 28, 28)).numpy()
        #img_flat = img_square.flatten()
        if self.transform is not None:
            img_square = self.transform(img_square)
        return img_square, label


def load_dataset(args):
    SIZE = args.size
    batch_size = args.bs

    # Load the Perceval Quest splits from the Merlin dataset helper
    X_train_raw, y_train_raw, _ = mnist_digits.get_data_train_percevalquest()
    X_val_raw, y_val_raw, _ = mnist_digits.get_data_test_percevalquest()

    # Crop a square at the centre of each image (SIZE x SIZE) and flatten it
    transform = TransformCenter(size=SIZE)
    X_train_flat = np.stack([transform(img).numpy() for img in X_train_raw]).astype(np.float32)
    X_val_flat = np.stack([transform(img).numpy() for img in X_val_raw]).astype(np.float32)

    y_train = np.asarray(y_train_raw, dtype=np.int64)
    y_val = np.asarray(y_val_raw, dtype=np.int64)

    # Feature-wise scaling (keeps behaviour identical to previous implementation)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)

    train_tensor = torch.from_numpy(X_train_scaled).float()
    val_tensor = torch.from_numpy(X_val_scaled).float()
    train_dataset = TensorDataset(train_tensor, torch.from_numpy(y_train))
    val_dataset = TensorDataset(val_tensor, torch.from_numpy(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    INPUT_SIZE = SIZE * SIZE
    OUTPUT_FEATURES = 10

    return X_train_scaled, X_val_scaled, y_train, y_val, train_loader, val_loader, INPUT_SIZE, OUTPUT_FEATURES

###############################
## Build the quantum circuit ##
###############################

def create_quantum_circuit(m, size = 400, frequency = 1):
    """Create quantum circuit with specified number of modes

        Args:
            m (int): number of modes
            size (int): size of the input data
            frequency (int): frequency of the repetition of the {encoding layers with input data in phase shifters; trainable generic interferometer}
    """

    # first trainable generic interferometer
    wl = pcvl.GenericInterferometer(m,
                                    lambda i: pcvl.BS(theta=pcvl.P(f"bs_1_{i}")) // pcvl.PS(pcvl.P(f"phase_1_{i}")) // \
                                              pcvl.BS(theta=pcvl.P(f"bs_2_{i}")) // pcvl.PS(pcvl.P(f"phase_2_{i}")),
                                    shape=pcvl.InterferometerShape.RECTANGLE)

    c = pcvl.Circuit(m)
    c.add(0, wl, merge=True)

    # f repetition of {encoding layers with input data in phase shifters; trainable generic interferometer}
    for f in range(frequency):
        c_var = pcvl.Circuit(m)
        for i in range(size):
            px = pcvl.P(f"px-{f}-{i + 1}")
            c_var.add(i%m, pcvl.PS(px))
        print(c_var)
        c.add(0, c_var, merge=True)
        wr = pcvl.GenericInterferometer(m,
                                        lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"phase_3_{i}")) // \
                                                  pcvl.BS() // pcvl.PS(pcvl.P(f"phase_4_{i}")),
                                        shape=pcvl.InterferometerShape.RECTANGLE)



        c.add(0, wr, merge=True)

    return c

## multply the input by either learnable weights or scalars ##
class ScaleLayer(nn.Module):
    def __init__(self, dim, scale_type = "learned"):
        super(ScaleLayer, self).__init__()
        # Create a single learnable parameter (initialized to 1.0 by default)
        # Caution: MerLin already mutltiplies by pi
        if scale_type == "learned":
            self.scale = nn.Parameter(torch.rand(dim))
        elif scale_type == "2pi":
            self.scale = torch.full((dim,), 2 * torch.pi)
        elif scale_type == "pi":
            self.scale = torch.full((dim,), torch.pi)
        elif scale_type == "1":
            self.scale = torch.full((dim,), 1)
        #print(f"SELF.SCALE: {self.scale.shape}")

    def forward(self, x):
        # Element-wise multiplication of each input element by the learned scale
        return x * self.scale

###################################
## Model training and evaluation ##
###################################

def train_model(model, train_loader, val_loader, num_epochs = 25, lr=0.01, frequency = 1, quantum = False):
    # train classical baseline
    criterion = nn.CrossEntropyLoss()
    # Betas from the ablation study
    optimizer = torch.optim.Adam(model.parameters(),lr = lr, betas=(0.8, 0.999))
    all_losses = []
    all_test_losses = []
    all_train_accuracies = []
    all_val_accuracies = []
    best_val_acc = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_test_loss = 0
        correct = 0
        total = 0
        train_acc = 0
        for batch_X, batch_y in tqdm(train_loader):
            # Forward pass
            batch_X = batch_X.repeat(1, 1, frequency)
            if quantum:
                batch_X = batch_X/torch.pi
            outputs = model(batch_X.squeeze(0).float())
            loss = criterion(outputs, batch_y) #.view(-1, 1).float()
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            # Update tqdm with current metrics
            current_accuracy = 100 * correct / total
            train_acc += current_accuracy
        model.eval()
        correct_test = 0
        total_test = 0
        val_acc = 0
        #for batch_X, batch_y in val_loader:
        for batch_X, batch_y in tqdm(val_loader):
            batch_X = batch_X.repeat(1, 1, frequency)
            if quantum:
                batch_X = batch_X/torch.pi
            outputs = model(batch_X.squeeze(0).float())
            test_loss = criterion(outputs, batch_y) #.view(-1, 1).float()
            total_test_loss += test_loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_test += batch_y.size(0)
            correct_test += (predicted == batch_y).sum().item()

            current_test_accuracy = 100 * correct_test / total_test
            val_acc += current_test_accuracy
        val_acc_epoch = val_acc / len(val_loader)
        train_acc_epoch = train_acc / len(train_loader)
        if val_acc_epoch > best_val_acc:
            best_val_acc = val_acc_epoch
        # Print progress
        avg_loss = total_loss / len(train_loader)
        avg_test_loss = total_test_loss / len(val_loader)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train loss: {avg_loss:.4f}, Test loss: {avg_test_loss:.4f}, Train acc: {train_acc_epoch:.4f}, Test acc: {val_acc_epoch:.4f}, Best val acc: {best_val_acc:.4f}')

        all_losses.append(avg_loss)
        all_test_losses.append(avg_test_loss)
        all_train_accuracies.append(train_acc_epoch)
        all_val_accuracies.append(val_acc_epoch)

    return all_losses, all_test_losses, best_val_acc, all_train_accuracies, all_val_accuracies

# count paramaeters in a model (nn.Module)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#############################
### results visualisation ###
#############################

## here, we want to visualize the scale parameters of the encoding function ##
def visualize_scale_parameters(scale_layer):
    """
        Display scale parameters of the encoding layers.

        Args:
            scale_layer (nn.Module): encoding layer
        """
    # Get the scale parameter data as a numpy array
    scale_data = scale_layer.scale.data.cpu().numpy()

    # For a single scale parameter
    if scale_data.size == 1:
        print(f"Learned scale parameter: {scale_data.item():.4f}")

    # For a 1D array of parameters (e.g., per feature)
    elif len(scale_data.shape) == 1 or (
            len(scale_data.shape) > 1 and np.prod(scale_data.shape) == max(scale_data.shape)):
        # Reshape to 1D if necessary
        scale_data = scale_data.flatten()

        plt.figure(figsize=(10, 6))

        # Option 1: Bar plot
        plt.subplot(2, 1, 1)
        plt.bar(range(len(scale_data)), scale_data)
        plt.title('Learned Scale Parameters')
        plt.xlabel('Parameter Index')
        plt.ylabel('Value')

        # Option 2: Heatmap (1D version)
        plt.subplot(2, 1, 2)
        sns.heatmap(scale_data.reshape(1, -1), cmap='viridis', annot=True if len(scale_data) < 20 else False)
        plt.title('Scale Parameters Heatmap')
        plt.xlabel('Parameter Index')

        plt.tight_layout()
        plt.savefig('scale_parameters.png')
        plt.show()

## display the confusion matrices for the 2 models ##
def display_confusion_matrices(model1, model2, val_loader, class_names=None, device='cuda'):
    """
    Display confusion matrices for two trained models using a validation data loader.

    Args:
        3 models : 3 PyTorch model
        val_loader: PyTorch DataLoader containing validation data
        class_names: List of class names (optional)
        device: Device to run inference on ('cuda' or 'cpu')
    """
    # Set models to evaluation mode
    model1.eval()
    model2.eval()

    # Move models to the appropriate device
    model1 = model1.to(device)
    model2 = model2.to(device)

    # Initialize lists to store predictions and ground truth
    all_preds1 = []
    all_preds2 = []
    all_targets = []

    # Disable gradient computation for inference
    with torch.no_grad():
        for inputs, targets in val_loader:

            targets = targets.to(device)

            # Get predictions from both models
            outputs1 = model1(inputs.squeeze(1).float())
            outputs2 = model2(inputs.squeeze(1).float())

            # Convert outputs to class predictions
            _, preds1 = torch.max(outputs1, 1)
            _, preds2 = torch.max(outputs2, 1)

            # Append batch predictions and targets to lists
            all_preds1.extend(preds1.cpu().numpy())
            all_preds2.extend(preds2.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds1 = np.array(all_preds1)
    all_preds2 = np.array(all_preds2)
    all_targets = np.array(all_targets)

    # Compute confusion matrices
    cm1 = confusion_matrix(all_targets, all_preds1)
    cm2 = confusion_matrix(all_targets, all_preds2)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Display confusion matrices
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=class_names)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=class_names)

    disp1.plot(ax=ax1, cmap='coolwarm', values_format='d')
    disp2.plot(ax=ax2, cmap='coolwarm', values_format='d')

    # Set titles
    ax1.set_title('Confusion Matrix - quantum Trained model')
    ax2.set_title('Confusion Matrix - classical Trained model')

    # Add overall accuracy to the titles
    acc1 = np.sum(np.diag(cm1)) / np.sum(cm1)
    acc2 = np.sum(np.diag(cm2)) / np.sum(cm2)

    ax1.set_xlabel(f'Predicted Label\nAccuracy: {acc1:.4f}')
    ax2.set_xlabel(f'Predicted Label\nAccuracy: {acc2:.4f}')

    plt.tight_layout()
    #plt.savefig(f'./results/CM-h-{hidden_dim}-m-{modes}.png')
    plt.show()

    return cm1, cm2

## extract features from a trained model for tSNE analysis ##
def extract_features(model, dataloader, device='cuda'):
    """
        Extract features using a trained model.

        Args:
            models : PyTorch model from which to extract features
            dataloader: dataloader containing training/validation data
            device: Device to run inference on ('cuda' or 'cpu')

        Returns:
            features: TorchTensor
            labels: TorchTensor
    """
    model.eval()  # Set the model to evaluation mode
    features = []
    labels = []

    with torch.no_grad():
        for data, label in dataloader:
            BS = data.shape[0]
            data = data.reshape(BS,-1).to(device)
            #print(f"\nData = {data}")
            output = model(data.float())
            features.append(output.cpu())  # Move to CPU for compatibility
            labels.extend(label.cpu().numpy())

    features = torch.cat(features, dim=0).numpy()
    labels = torch.tensor(labels).numpy()
    return features, labels


## display the tSNE plots for 2 models and dataloader ##
def display_tsne(model1, model2, val_loader,modes, device='cpu'):
    """
    Display the 3 tSNE for 3 trained models using a validation data loader.

    Args:
        the 2 models: 2 PyTorch models we want to compare
        val_loader: PyTorch DataLoader containing validation data
        modes: number used for the quantum model (for the Figure title)
        device: Device to run inference on ('cuda' or 'cpu')
    """
    # Set models to evaluation mode
    model1.eval()
    model2.eval()

    # Move models to the appropriate device
    model1 = model1.to(device)
    model2 = model2.to(device)

    # get the features and compute tSNE for each model
    features_1, labels_1 = extract_features(model1, val_loader, device=device)
    tsne = TSNE(n_components=2, random_state=42)
    features_2d_1 = tsne.fit_transform(features_1)

    features_2, labels_2 = extract_features(model2, val_loader, device=device)
    tsne = TSNE(n_components=2, random_state=42)
    features_2d_2 = tsne.fit_transform(features_2)


    # Display the tSNE plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    num_classes = 10
    for class_idx in range(num_classes):
        ax1.scatter(features_2d_1[labels_1 == class_idx, 0], features_2d_1[labels_1 == class_idx, 1],
                    label=f'Digit {class_idx}', alpha=0.6)
        ax1.set_xlabel('t-SNE Dim 1')
        ax1.set_ylabel('t-SNE Dim 2')
        ax1.legend()
        ax2.scatter(features_2d_2[labels_2 == class_idx, 0], features_2d_2[labels_2 == class_idx, 1],
                    label=f'Digit {class_idx}', alpha=0.6)
        ax2.set_xlabel('t-SNE Dim 1')
        ax2.set_ylabel('t-SNE Dim 2')
        ax2.legend()

    # Set titles
    ax1.set_title('tSNE - quantum kernel')
    ax2.set_title('tSNE - classical kernel model')
    plt.tight_layout()
    plt.savefig(f'tSNE-m-{modes}.png')
    #plt.show()

    return "done"

def save_experiment_results(results, filename='photonic_qNN_results.json'):
    """
    Append experiment results to a JSON file.

    Args:
        results (dict): Dictionary containing experiment results (with float values)
        filename (str): Path to the JSON file to store results
    """
    # Check if file exists and load existing data
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as file:
                all_results = json.load(file)
        except json.JSONDecodeError:
            # Handle case where file exists but is empty or corrupted
            all_results = []
    else:
        all_results = []

    # Append new results
    all_results.append(results)

    # Write updated data back to file
    with open(filename, 'w') as file:
        json.dump(all_results, file, indent=4)

    return len(all_results)



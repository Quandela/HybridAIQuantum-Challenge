#!/usr/bin/env python3
import os
import torch
import perceval as pcvl
import perceval.providers.scaleway as scw
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import pickle
import yaml
import sys


# Import configuration, models, training routines, and utilities.
from utils import load_config, create_scaleway_session
from boson_sampler import BosonSampler, VariationalBosonSampler, ConvolutionalInterferometerBuilder, TriangularInterferometerBuilder, RectangularInterferometerBuilder, PdfInterferometerBuilder, BaseInterferometerBuilder
from model import MnistModel, HybridMnistModel
from training import fit, evaluate
from utils import MNIST_partial, get_dataloader, plot_training_metrics, compute_downsample_shape, pca_downsample

# For classification report and LaTeX generation (from main_classic.py)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
import seaborn as sns
from datetime import datetime
import pandas as pd


# Import dataset and utility functions
from utils import MNIST_partial, get_dataloader, plot_training_metrics, accuracy

from pathlib import Path
script_dir = Path(__file__).parent
DATA_PATH= (script_dir / ".." / ".." / "data").resolve()

# Path to PCA model if needed.
# Get the directory of this file (training.py)
base_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level (assuming your project structure has 'src' and 'data' as siblings)
data_dir = os.path.join(base_dir, "..", "data")
pca_model_path = os.path.join("./data/pca_model.pkl")


#######################
# Configuration Loader
#######################
def load_config(config_path="yaml_default_config.yaml"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_full_path = os.path.join(base_dir, config_path)
    with open(config_full_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

########################
# PCA Downsampling Helper
########################
def prepare_pca_data(dataset, n_components=126):
    """
    Given a MNIST_partial dataset, flatten each image and return an array of shape (N, 784)
    and labels as a numpy array.
    """
    data_list = []
    labels_list = []
    for img, label in dataset:
        # Each image is shape [1,28,28]. Flatten to 784.
        data_list.append(img.view(-1).numpy())
        labels_list.append(label)
    X = np.array(data_list)
    y = np.array(labels_list)
    return X, y

###########################
# Define the Classical Model
###########################
class ClassicPCAClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=10):
        super(ClassicPCAClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # x is assumed to be of shape [B, input_dim]
        out = self.model(x)
        return out

###########################
# Custom Dataset for PCA features
###########################
class PCADataset(Dataset):
    def __init__(self, X, y):
        """
        X: numpy array of shape [N, n_components]
        y: numpy array of shape [N]
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


###########################
# Visualization Functions
###########################

def plot_pca_components(pca, num_components=16, result_dir="."):
    """Visualize the top PCA components"""
    components = pca.components_
    
    # Plot the first num_components components as 28x28 images
    n_cols = 4
    n_rows = (num_components + n_cols - 1) // n_cols
    
    plt.figure(figsize=(12, 3*n_rows))
    for i in range(min(num_components, components.shape[0])):
        plt.subplot(n_rows, n_cols, i+1)
        component = components[i].reshape(28, 28)
        plt.imshow(component, cmap='viridis')
        plt.title(f"Component {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{result_dir}/pca_components.png", dpi=300)
    plt.close()

def plot_digit_reconstructions(pca, X_test, y_test, num_examples=10, result_dir="."):
    """Visualize original digits and their PCA reconstructions"""
    plt.figure(figsize=(15, 3*num_examples))
    
    for i in range(num_examples):
        # Original image
        plt.subplot(num_examples, 2, 2*i+1)
        original = X_test[i].reshape(28, 28)
        plt.imshow(original, cmap='gray')
        plt.title(f"Original (Digit {y_test[i]})")
        plt.axis('off')
        
        # Reconstructed image
        plt.subplot(num_examples, 2, 2*i+2)
        reconstructed = pca.inverse_transform(pca.transform([X_test[i]]))[0].reshape(28, 28)
        plt.imshow(reconstructed, cmap='gray')
        plt.title(f"PCA Reconstruction")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{result_dir}/digit_reconstructions.png", dpi=300)
    plt.close()

def plot_variance_explained(pca, result_dir="."):
    """Plot the cumulative explained variance ratio"""
    plt.figure(figsize=(10, 6))
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(cumulative_variance, 'b-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of PCA Components')
    plt.grid(True)
    
    # Add markers for specific variance thresholds
    thresholds = [0.7, 0.8, 0.9, 0.95, 0.99]
    for threshold in thresholds:
        components_needed = np.argmax(cumulative_variance >= threshold) + 1
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.3)
        plt.axvline(x=components_needed, color='g', linestyle='--', alpha=0.3)
        plt.annotate(f"{threshold*100:.0f}%: {components_needed} components", 
                    xy=(components_needed+5, threshold),
                    fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{result_dir}/variance_explained.png", dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, result_dir="."):
    """Generate and save a confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{result_dir}/confusion_matrix.png", dpi=300)
    plt.close()

def plot_digit_collage(dataset, result_dir="."):
    """Create a collage of sample digits from each class"""
    # Get samples from each class
    digit_samples = {i: [] for i in range(10)}
    for img, label in dataset:
        if len(digit_samples[label]) < 5:  # Get 5 samples per digit
            digit_samples[label].append(img)
        if all(len(samples) >= 5 for samples in digit_samples.values()):
            break
    
    # Plot collage
    plt.figure(figsize=(12, 6))
    for digit in range(10):
        for i, img in enumerate(digit_samples[digit]):
            plt.subplot(10, 5, 5*digit + i + 1)
            plt.imshow(img.squeeze(0), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title(f"Digit {digit}")
    
    plt.tight_layout()
    plt.savefig(f"{result_dir}/digit_collage.png", dpi=300)
    plt.close()





def plot_misclassified_predictions_classic(model, val_loader, device, result_dir=".", pca=None, max_images=50):
    """
    Create a collage showing only misclassified validation images.
    If a PCA model is provided and the inputs are PCA features,
    it uses a vectorized inverse_transform to reconstruct images.
    
    Parameters:
      - model: the trained classifier.
      - val_loader: DataLoader yielding PCA-transformed data.
      - device: computation device.
      - result_dir: where to save the plot.
      - pca: the fitted PCA object (optional).
      - max_images: maximum number of misclassified images to show.
    """
    model.eval()
    all_imgs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_imgs.append(imgs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    all_imgs = torch.cat(all_imgs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Get indices of misclassified samples
    mis_idx = (all_preds != all_labels).nonzero(as_tuple=True)[0]
    if mis_idx.numel() == 0:
        print("No misclassified images to plot.")
        return
    
    # Limit the number of misclassified images if needed
    if mis_idx.numel() > max_images:
        mis_idx = mis_idx[:max_images]
    
    mis_imgs = all_imgs[mis_idx]
    mis_preds = all_preds[mis_idx]
    mis_labels = all_labels[mis_idx]
    
    # If the images are PCA features (i.e. 2D tensors) then reconstruct them in one go.
    if pca is not None and mis_imgs.ndimension() == 2 and mis_imgs.shape[1] == pca.n_components_:
        imgs_np = mis_imgs.numpy()  # shape: (N, n_components)
        reconstructed = pca.inverse_transform(imgs_np)  # shape: (N, 784)
        reconstructed_imgs = reconstructed.reshape(-1, 28, 28)
    else:
        # Otherwise assume the images are already in image format.
        reconstructed_imgs = [img.squeeze(0).numpy() for img in mis_imgs]
    
    N = len(reconstructed_imgs)
    n_cols = 5
    n_rows = (N + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    axes = axes.flatten()
    for i in range(N):
        ax = axes[i]
        ax.imshow(reconstructed_imgs[i], cmap='gray')
        ax.axis('off')
        ax.set_title(f"T:{mis_labels[i].item()} P:{mis_preds[i].item()}", fontsize=8)
    for j in range(N, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(f"{result_dir}/misclassified_predictions.png", dpi=300)
    plt.close()


def generate_latex_table(train_metrics, val_metrics, result_dir="."):
    """Generate LaTeX table with model performance metrics"""
    latex_content = r"""
\begin{table}[h]
\centering
\caption{Classical PCA-based Model Performance}
\begin{tabular}{lccc}
\hline
\textbf{Dataset} & \textbf{Accuracy (\%)} & \textbf{F1 Score (Macro)} & \textbf{F1 Score (Weighted)} \\
\hline
Training & """ + f"{train_metrics['accuracy']*100:.2f}" + r""" & """ + f"{train_metrics['f1_macro']:.4f}" + r""" & """ + f"{train_metrics['f1_weighted']:.4f}" + r""" \\
Validation & """ + f"{val_metrics['accuracy']*100:.2f}" + r""" & """ + f"{val_metrics['f1_macro']:.4f}" + r""" & """ + f"{val_metrics['f1_weighted']:.4f}" + r""" \\
\hline
\end{tabular}
\end{table}
"""
    
    with open(f"{result_dir}/performance_table.tex", "w") as f:
        f.write(latex_content)
    
    return latex_content

def generate_latex_report(train_metrics, val_metrics, pca_metrics, result_dir="."):
    """Generate a complete LaTeX report"""
    latex_content = r"""

We reduced the dimensionality of the MNIST dataset using PCA to """ + f"{pca_metrics['n_components']}" + r""" components,
which explains """ + f"{pca_metrics['explained_variance']*100:.2f}%" + r""" of the variance in the data.
A simple neural network classifier was then trained on these reduced features.

PCA reduced the original 784-dimensional space (28×28 images) to """ + f"{pca_metrics['n_components']}" + r""" dimensions
while preserving """ + f"{pca_metrics['explained_variance']*100:.2f}%" + r""" of the variance.

The classical PCA-based approach achieves """ + f"{val_metrics['accuracy']*100:.2f}%" + r""" accuracy on the validation set.

\end{document}
"""
    
    with open(f"{result_dir}/full_report.tex", "w") as f:
        f.write(latex_content)
    
    return latex_content


# Updated plotting function for misclassified predictions.
def plot_misclassified_predictions(model, val_loader, device, result_dir=".", pca=None, bs=None, cfg=None):
    model.eval()
    all_imgs, all_preds, all_labels = [], [], []
    # Determine target downsample shape if quantum embedding is needed.
    target_shape = compute_downsample_shape(bs.nb_parameters) if (bs is not None and cfg is not None) else None
    
    import torch.nn.functional as F
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            if model.embedding_size and bs is not None and cfg is not None:
                batch_embeddings = []
                for i in range(imgs.size(0)):
                    img = imgs[i]
                    if cfg["quantum"].get("downsample_method", "bilinear") == "pca":
                        with open(pca_model_path, "rb") as f:
                            pca_model = pickle.load(f)
                        img_resized = pca_downsample(img, pca_model)
                    else:
                        img_expanded = img.unsqueeze(0)
                        img_resized = F.interpolate(img_expanded, size=target_shape, mode='bilinear', align_corners=False)
                        img_resized = img_resized.squeeze(0)
                    emb = bs.embed(img_resized, bs.n_samples)
                    batch_embeddings.append(emb)
                emb_tensor = torch.stack(batch_embeddings)
                outputs = model(imgs, emb_tensor)
            else:
                outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_imgs.append(imgs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    all_imgs = torch.cat(all_imgs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Identify misclassified images.
    mis_idx = (all_preds != all_labels).nonzero(as_tuple=True)[0]
    if mis_idx.numel() == 0:
        print("No misclassified images to plot.")
        return
    # Limit number of misclassified images.
    max_images = 50
    if mis_idx.numel() > max_images:
        mis_idx = mis_idx[:max_images]
    mis_imgs = all_imgs[mis_idx]
    mis_preds = all_preds[mis_idx]
    mis_labels = all_labels[mis_idx]
    
    # Generate collage plot.
    import matplotlib.pyplot as plt
    N = mis_imgs.size(0)
    n_cols = 5
    n_rows = (N + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    axes = axes.flatten()
    for i in range(N):
        ax = axes[i]
        # Assuming images are in [1,28,28] format.
        ax.imshow(mis_imgs[i].squeeze(0), cmap='gray')
        ax.axis('off')
        ax.set_title(f"T:{mis_labels[i].item()} P:{mis_preds[i].item()}", fontsize=8)
    for j in range(N, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    collage_path = os.path.join(result_dir, "misclassified_predictions.png")
    plt.savefig(collage_path, dpi=300)
    plt.close()
    print("Misclassified predictions collage saved to", collage_path)







def setup_result_directory(prefix="hybrid_run"):
    """Create a timestamped directory to save all outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("results", f"{prefix}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def collect_predictions(model, val_loader, device, bs, cfg):
    """
    Iterate over the validation set to collect true and predicted labels.
    For hybrid models, compute quantum embeddings per image (downsampled as configured).
    """
    model.eval()
    all_preds = []
    all_labels = []
    # Determine target downsample shape.
    target_shape = compute_downsample_shape(bs.nb_parameters)

    
    import torch.nn.functional as F
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch  # images shape: [B, 1, 28, 28]
            images, labels = images.to(device), labels.to(device)
            if model.embedding_size:
                batch_embeddings = []
                for i in range(images.size(0)):
                    img = images[i]  # [1,28,28]
                    # Downsample image if necessary:
                    if cfg["quantum"].get("downsample_method", "bilinear") == "pca":
                        with open(pca_model_path, "rb") as f:
                            pca_model = pickle.load(f)
                        img_resized = pca_downsample(img, pca_model)
                    else:
                        img_expanded = img.unsqueeze(0)  # [1,1,28,28]
                        img_resized = F.interpolate(img_expanded, size=target_shape, mode='bilinear', align_corners=False)
                        img_resized = img_resized.squeeze(0)
                    # Compute embedding (using standard embed here; adaptive option can be added similarly)
                    emb = bs.embed(img_resized, bs.n_samples)
                    batch_embeddings.append(emb)
                emb_tensor = torch.stack(batch_embeddings)
                outputs = model(images, emb_tensor)
            else:
                outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

def hybrid_main():
    # 1. Load configuration.
    cfg = load_config()
    print("Loaded configuration:", cfg)

    # 2. Create a results directory.
    result_dir = setup_result_directory(prefix="hybrid_run")
    print(f"Results will be saved to: {result_dir}")

    # 3. Create Scaleway session if remote backend is chosen.
    session = None

    print("Proceeding with local simulation backend.")

    # 4. Select circuit builder.
    circuit_type = cfg["quantum"].get("circuit_type", "triangular")
    if circuit_type == "pdf":
        builder = PdfInterferometerBuilder()
    elif circuit_type == "base":
        builder = BaseInterferometerBuilder()
    elif circuit_type == "triangular":
        builder = TriangularInterferometerBuilder()
    elif circuit_type == "rectangular":
        builder = RectangularInterferometerBuilder()
    elif circuit_type == "convolutional":
        import numpy as np
        filterA = np.random.rand(6, 5)
        filterB = np.random.rand(6, 5)
        builder = ConvolutionalInterferometerBuilder(filterA, filterB, image_size=(28,28))
    else:
        raise ValueError(f"Unsupported circuit type: {circuit_type}")

    # 5. Initialize BosonSampler.
    if cfg["quantum"].get("variational", False):
        bs = VariationalBosonSampler(
            m=cfg["quantum"]["n_modes"],
            n=cfg["quantum"]["n_photons"],
            postselect=cfg["quantum"]["postselect"],
            backend=cfg["quantum"]["local_backend"],
            session=session,
            builder=builder,
            cache_enabled=cfg["quantum"].get("cache_enabled", False),
            cache_directory=cfg["quantum"].get("cache_directory", "results/cache")
        )
    else:
        bs = BosonSampler(
            m=cfg["quantum"]["n_modes"],
            n=cfg["quantum"]["n_photons"],
            postselect=cfg["quantum"]["postselect"],
            backend=cfg["quantum"]["local_backend"],
            session=session,
            builder=builder,
            cache_enabled=cfg["quantum"].get("cache_enabled", False),
            cache_directory=cfg["quantum"].get("cache_directory", "results/cache")
        )
    bs.n_samples = cfg["quantum"]["n_samples"]
    print(f"BosonSampler: m={bs.m}, n={bs.n}, nb_parameters={bs.nb_parameters}, embedding_size={bs.embedding_size}")
    # Display the circuit layout.
    pcvl.pdisplay(bs.create_circuit())

    # 6. Load MNIST dataset.
    train_dataset = MNIST_partial(data=DATA_PATH, split="train")
    val_dataset = MNIST_partial(data=DATA_PATH, split="val")
    train_loader = get_dataloader(train_dataset, batch_size=cfg["data"]["batch_size"], shuffle=cfg["data"]["shuffle"])
    val_loader = get_dataloader(val_dataset, batch_size=cfg["data"]["batch_size"], shuffle=False)
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # 7. Set device.
    device = "cuda" if (cfg["training"]["device"]=="auto" and torch.cuda.is_available()) else "cpu"
    print("Using device:", device)

    # 8. Initialize model.
    embedding_size = bs.embedding_size if cfg["model"]["type"]=="hybrid" else 0
    if cfg["model"]["type"] == "hybrid":
        model = HybridMnistModel(device=device, embedding_size=embedding_size)
    else:
        model = MnistModel(device=device, embedding_size=0)
    model = model.to(device)
    print(f"Initialized {'quantum-enhanced' if model.embedding_size else 'classical'} model on {device}.")

    # 9. Setup optimizer.
    optimizer_name = cfg["training"]["optimizer"].lower()
    if cfg["quantum"].get("variational", False):
        params_to_opt = list(model.parameters()) + [bs.params]
    else:
        params_to_opt = model.parameters()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(params_to_opt, lr=cfg["training"]["learning_rate"])
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(params_to_opt, lr=cfg["training"]["learning_rate"])
    else:
        raise ValueError("Unsupported optimizer specified in config.")

    # 10. Train the model.
    print(f"Starting training for {cfg['training']['epochs']} epochs (lr = {cfg['training']['learning_rate']})...")
    history = fit(
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["learning_rate"],
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        bs=bs,
        optimizer=optimizer,
        cfg=cfg
    )
    print("Training completed.")

    # 11. Evaluate the model and collect predictions.
    final_metrics = evaluate(model, val_loader, bs, cfg)
    print(f"Final Validation Loss: {final_metrics['val_loss']:.4f}, Accuracy: {final_metrics['val_acc']:.4f}")

    true_labels, pred_labels = collect_predictions(model, val_loader, device, bs, cfg)
    # Save classification report.
    report = classification_report(true_labels, pred_labels, digits=4)
    report_path = os.path.join(result_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # 12. Generate additional plots and LaTeX reports.
    # Plot training curves (if your fit() function internally plots and saves, this step may be redundant).
    # Here, we assume plot_training_metrics() saves a file "training_curves.png" in the current directory.
    plot_training_metrics([h["val_acc"] for h in history], [h["val_acc"] for h in history],
                            [h["val_loss"] for h in history], [h["val_loss"] for h in history])
    os.replace("training_curves.png", os.path.join(result_dir, "training_curves.png"))
    print("Training curves saved.")

    # Generate and save confusion matrix plot.
    plot_confusion_matrix(true_labels, pred_labels, result_dir)
    print("Confusion matrix plot saved.")

    # Generate and save misclassified predictions collage.
    plot_misclassified_predictions(model, val_loader, device, result_dir, pca=None, bs=bs, cfg=cfg)
    print("Misclassified predictions collage saved.")

    # Optionally, generate LaTeX table and full report.
    try:
        from utils import generate_latex_table, generate_latex_report
        # For demonstration, we use final validation metrics as both training and validation metrics.
        latex_table = generate_latex_table(
            {"accuracy": final_metrics["val_acc"], "f1_macro": 0.0, "f1_weighted": 0.0},
            {"accuracy": final_metrics["val_acc"], "f1_macro": 0.0, "f1_weighted": 0.0},
            result_dir
        )
        latex_report = generate_latex_report(
            {"accuracy": final_metrics["val_acc"], "f1_macro": 0.0, "f1_weighted": 0.0},
            {"accuracy": final_metrics["val_acc"], "f1_macro": 0.0, "f1_weighted": 0.0},
            {},  # No PCA metrics in the hybrid model case
            result_dir
        )
        print("LaTeX table and report generated.")
    except Exception as e:
        print("Error generating LaTeX reports:", e)

    # 13. Save a summary of results.
    summary = {
        'final_val_loss': final_metrics['val_loss'],
        'final_val_accuracy': final_metrics['val_acc'],
        'training_history': history,
        'classification_report': report
    }
    summary_path = os.path.join(result_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {summary_path}")



###########################
# Main training function for the classical baseline
###########################
def main_classic():
    # Create result directory
    result_dir = setup_result_directory(prefix="classical_run")
    print(f"Results will be saved to: {result_dir}")

    # Load configuration
    cfg = load_config()
    print("Loaded configuration:", cfg)
    
    # For the classical baseline we can optionally use a different batch size.
    # For example, set it to 30 (or 1 if you want strictly sequential processing).
    batch_size = cfg["data"].get("batch_size", 30)
    
    # Load the MNIST dataset (CSV version) using MNIST_partial
    train_dataset = MNIST_partial(data=DATA_PATH, split="train")
    val_dataset   = MNIST_partial(data=DATA_PATH, split="val")
    
    # Create a digit collage visualization
    plot_digit_collage(train_dataset, result_dir)

    # Prepare data for PCA: flatten each image (each image: [1,28,28] -> 784,)
    print("Preparing data for PCA...")
    X_train, y_train = prepare_pca_data(train_dataset, n_components=784)
    X_val, y_val     = prepare_pca_data(val_dataset, n_components=784)
    
    # Define PCA parameters:
    # For our purpose, we choose n_components equal to 126, which corresponds to:
    # nb_parameters = m*(m-1) - (m//2) for m=12 (i.e. 12*11 - 6 = 126)
    m = cfg["quantum"]["n_modes"]
    n_components = m * (m - 1) - (m // 2)  # Formula for parameter count in triangular mesh
    print(f"Using PCA with {n_components} components to match quantum parameter count")
    
    # Fit PCA on the training data
    print("Fitting PCA on training data...")
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca   = pca.transform(X_val)
    

    # Calculate the explained variance with the chosen components
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"PCA with {n_components} components explains {explained_variance*100:.2f}% of variance")
    
    # Store PCA metrics for reporting
    pca_metrics = {
        'n_components': n_components,
        'explained_variance': explained_variance
    }
    
    # Create PCA visualizations
    plot_pca_components(pca, num_components=16, result_dir=result_dir)
    plot_digit_reconstructions(pca, X_val, y_val, num_examples=5, result_dir=result_dir)
    plot_variance_explained(pca, result_dir=result_dir)


    # Save the PCA model to a pickle file so it can be reused later
    # Ensure the data folder exists
    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    pca_filename = os.path.join(data_folder, "pca_model.pkl")
    with open(pca_filename, "wb") as f:
        pickle.dump(pca, f)
    print(f"PCA model saved to {pca_filename}")
    
    # Create datasets and dataloaders from the PCA-transformed data
    train_dataset_pca = PCADataset(X_train_pca, y_train)
    val_dataset_pca   = PCADataset(X_val_pca, y_val)
    train_loader = DataLoader(train_dataset_pca, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset_pca, batch_size=batch_size, shuffle=False)
    
    # Determine device: Use CUDA if available (for faster training)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    # Build a simple classifier that takes PCA features as input
    hidden_dim = cfg["model"].get("hidden_dim", 64) #n_components
    model = ClassicPCAClassifier(input_dim=n_components, hidden_dim=hidden_dim, num_classes=10)
    model = model.to(device)
    
    # Set up optimizer and loss function
    learning_rate = cfg["training"].get("learning_rate", 0.001)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    epochs = cfg["training"].get("epochs", 5)
    
    # Training loop variables to record metrics
    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []
    
    # Training loop
    print("Starting training of classical baseline...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        num_samples = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Training)"):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            num_samples += inputs.size(0)
        
        epoch_loss = running_loss / num_samples
        epoch_acc  = running_corrects / num_samples
        train_loss_list.append(epoch_loss)
        train_acc_list.append(epoch_acc)


        # Calculate F1 scores
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, F1 (macro): {f1_macro:.4f}")
        
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_preds = []
        val_labels = []
        num_val_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Validation)"):
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels).item()
                num_val_samples += inputs.size(0)

                # Collect predictions for metrics
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())


        epoch_val_loss = val_loss / num_val_samples
        epoch_val_acc  = val_corrects / num_val_samples
        val_loss_list.append(epoch_val_loss)
        val_acc_list.append(epoch_val_acc)
        
        # Calculate validation F1 scores
        val_f1_macro = f1_score(val_labels, val_preds, average='macro')
        val_f1_weighted = f1_score(val_labels, val_preds, average='weighted')
        
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")
    




# Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_acc_list, label='Training')
    plt.plot(range(1, epochs+1), val_acc_list, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_loss_list, label='Training')
    plt.plot(range(1, epochs+1), val_loss_list, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{result_dir}/training_curves.png", dpi=300)
    plt.close()

     # Plot training curves
    plot_training_metrics(train_acc_list, val_acc_list, train_loss_list, val_loss_list)
    

    # Generate confusion matrix
    plot_confusion_matrix(val_labels, val_preds, result_dir)
    
    # Generate a collage of validation images with markers indicating prediction correctness
    plot_misclassified_predictions_classic(model, val_loader, device, result_dir, pca)

    # Generate detailed classification report
    report = classification_report(val_labels, val_preds, digits=4)
    with open(f"{result_dir}/classification_report.txt", "w") as f:
        f.write(report)
    
    # Store final metrics for the LaTeX report
    train_metrics = {
        'accuracy': epoch_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }
    
    val_metrics = {
        'accuracy': epoch_val_acc,
        'f1_macro': val_f1_macro, 
        'f1_weighted': val_f1_weighted
    }
    
    # Save model for later use
    torch.save(model.state_dict(), f"{result_dir}/classic_model.pt")
    
    # Generate LaTeX table and full report
    generate_latex_table(train_metrics, val_metrics, result_dir)
    generate_latex_report(train_metrics, val_metrics, pca_metrics, result_dir)
    
    # Save a summary of results
    summary = {
        'pca_components': int(n_components),
        'explained_variance': float(explained_variance),
        'final_train_accuracy': float(epoch_acc),
        'final_val_accuracy': float(epoch_val_acc),
        'final_train_f1_macro': float(f1_macro),
        'final_val_f1_macro': float(val_f1_macro),
        'train_loss': [float(x) for x in train_loss_list],
        'train_accuracy': [float(x) for x in train_acc_list],
        'val_loss': [float(x) for x in val_loss_list],
        'val_accuracy': [float(x) for x in val_acc_list]
    }
    
    with open(f"{result_dir}/summary.pkl", "wb") as f:
        pickle.dump(summary, f)
    

    import json
    summary_json = os.path.join(result_dir, "summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary metrics saved to {summary_json}")


    df_summary = pd.DataFrame({
        "Epoch": list(range(1, epochs+1)),
        "Train_Loss": train_loss_list,
        "Train_Accuracy": train_acc_list,
        "Val_Loss": val_loss_list,
        "Val_Accuracy": val_acc_list
    })
    df_summary.to_csv(os.path.join(result_dir, "training_history.csv"), index=False)
    print(f"Training history saved to {os.path.join(result_dir, 'training_history.csv')}")


    print(f"All outputs and visualizations saved to {result_dir}")
    print(f"Final validation accuracy: {epoch_val_acc:.4f}")
    print(f"Final validation F1 score (macro): {val_f1_macro:.4f}")
    print(f"LaTeX report generated at {result_dir}/full_report.tex")
    
    # Return the directory with all outputs
    return result_dir, summary


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]  # il primo argomento dopo il nome dello script
        if mode == "classic":
            main_classic()
        elif mode == "hybrid":
            hybrid_main()
    else:
        print("Usage: python main.py [classic|hybrid]")

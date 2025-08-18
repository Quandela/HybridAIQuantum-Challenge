import os
import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


#####################
#   DATA CONFIG     #
#####################

# previously src/config.py
import yaml
from decouple import config as decouple_config

def load_config(config_path="yaml_default_config.yaml"):
    """
    Load configuration from the YAML file.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_full_path = os.path.join(base_dir, config_path)
    with open(config_full_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def create_scaleway_session(cfg):
    """
    Create a Scaleway QaaS session based on the config and environment variables.
    Expects SCW_PROJECT_ID and SCW_SECRET_KEY to be set in the environment.
    """
    import perceval.providers.scaleway as scw
    # Get credentials from the .env file using python-decouple.
    proj_id = decouple_config("SCW_PROJECT_ID") # 
    token = decouple_config("SCW_SECRET_KEY") # 
    platform = cfg["quantum"].get("scaleway_platform", "sim:sampling:p100")
    if not proj_id or not token:
        raise RuntimeError("Scaleway credentials not found in environment.")
    session = scw.Session(
        project_id=proj_id,
        token=token,
        platform=platform,
        max_idle_duration_s=1200,
        max_duration_s=3600
    )
    return session
















#####################
#   DATA UTILS      #
#####################
class MNIST_partial(Dataset):
    def __init__(self, data="./data", transform=None, split="train"):
        """
        Load the reduced MNIST dataset from CSV.
        :param data: Path to folder containing train.csv and val.csv.
        :param transform: Optional transform function.
        :param split: "train" or "val"
        """
        # Compute the base directory (directory of this file)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # If 'data' is given as a relative path, make it absolute relative to this file:
        self.data_dir = os.path.join(base_dir, "..", data)
        self.transform = transform
        if split == "train":
            filename = os.path.join(self.data_dir, "train.csv")
        elif split == "val":
            filename = os.path.join(self.data_dir, "val.csv")
        else:
            raise AttributeError("split must be 'train' or 'val'")
        self.df = pd.read_csv(filename)

    def __len__(self):
        return len(self.df["image"])

    def __getitem__(self, idx):
        img_str = self.df["image"].iloc[idx]
        label = int(self.df["label"].iloc[idx])
        img_list = re.split(r",", img_str)
        img_list[0] = img_list[0][1:]
        img_list[-1] = img_list[-1][:-1]
        img_float = [float(el) for el in img_list]
        # Reshape to (1, 28, 28)
        img_tensor = torch.tensor(img_float).reshape(1, 28, 28)
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)
        return img_tensor, label

def get_dataloader(dataset, batch_size, shuffle):
    """Return a DataLoader for the given dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

#####################
# TRAINING UTILS    #
#####################
def plot_training_metrics(train_acc, val_acc, train_loss, val_loss):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    epochs = list(range(1, len(train_acc) + 1))
    axes[0].plot(epochs, train_acc, label="Training Accuracy")
    axes[0].plot(epochs, val_acc, label="Validation Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].legend()
    axes[1].plot(epochs, train_loss, label="Training Loss")
    axes[1].plot(epochs, val_loss, label="Validation Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.close()

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def compute_downsample_shape(nb_params, target_aspect=1.0):
    """
    Compute a downsample shape (height, width) such that height * width == nb_params if possible.
    If not, choose the factor pair that is as close as possible to a square (or the target aspect ratio).
    target_aspect=1.0 corresponds to a square.
    
    Parameters:
      nb_params: int - the desired total number of pixels (phase parameters)
      target_aspect: float - desired aspect ratio (width/height), default 1.0
      
    Returns:
      (height, width): tuple of ints representing the downsample shape.
    """
    from math import sqrt
    # Try to find factor pairs exactly.
    for h in range(int(sqrt(nb_params)), 0, -1):
        if nb_params % h == 0:
            w = nb_params // h
            #print(f"\n Downsample shape with nb params =  {nb_params} -> {h} x {w} \n")
            return h, w
    # If no factor pair exactly divides nb_params, default to a square shape.
    h = int(sqrt(nb_params))
    #print(f"\n Downsample shape with nb params =  {nb_params} -> {h} x {h} \n")
    return h, h




######################
# PCA Downsampling   #
######################


from sklearn.decomposition import PCA
import numpy as np
import torch

def pca_downsample(image_tensor, pca_model):
    """
    Downsample a 28x28 image using a pre-trained PCA model.
    
    Parameters:
      image_tensor: torch tensor of shape [1, 28, 28] (or [28,28])
      pca_model: a scikit-learn PCA model pre-trained on flattened MNIST images
      
    Returns:
      A torch tensor of shape [pca_model.n_components]
    """
    # Ensure the image is in 2D: if [1,28,28], remove the channel dimension.
    if image_tensor.dim() == 3 and image_tensor.shape[0] == 1:
        image_tensor = image_tensor.squeeze(0)
    # Flatten the image (28*28)
    img_flat = image_tensor.view(-1).cpu().numpy()
    # Project the flattened image onto the PCA components
    reduced = pca_model.transform([img_flat])[0]
    # Convert back to a torch tensor
    return torch.tensor(reduced, dtype=torch.float32)










import torch.fft

def fourier_encode(image_tensor, n_features=100):
    """
    Computes Fourier features from an image tensor.
    
    Parameters:
      image_tensor: A tensor of shape [1, 28, 28] (or [28,28]).
      n_features  : Number of Fourier features to extract.
    
    Returns:
      A 1D tensor of Fourier features (absolute values of the FFT).
    """
    # Ensure the image is 2D.
    if image_tensor.dim() == 3 and image_tensor.shape[0] == 1:
        image_tensor = image_tensor.squeeze(0)
    fft_result = torch.fft.fft2(image_tensor)
    fft_abs = torch.abs(fft_result).flatten()
    # Select the first n_features; you might also choose to sort or use a different selection method.
    return fft_abs[:n_features]






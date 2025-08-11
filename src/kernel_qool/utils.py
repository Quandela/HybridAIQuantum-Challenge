import os
import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset

class MNIST_partial(Dataset):
    def __init__(self, data="./data", transform=None, split="train"):
        """
        A minimal CSV-based MNIST loader.
        CSV must have columns: 'image' (a string "[0.1,0.2,...]") and 'label'.
        """
        self.data_dir = data
        self.transform = transform

        if split == 'train':
            filename = os.path.join(self.data_dir, 'train.csv')
        elif split == 'val':
            filename = os.path.join(self.data_dir, 'val.csv')
        else:
            raise AttributeError("split must be 'train' or 'val'")

        self.df = pd.read_csv(filename)

    def __len__(self):
        return len(self.df['image'])

    def __getitem__(self, idx):
        img_str = self.df['image'].iloc[idx]
        label   = self.df['label'].iloc[idx]
        # Convert from string e.g. "[0.0,0.1,0.2,...]" to list of floats
        img_list = re.split(r',', img_str)
        img_list[0]  = img_list[0].lstrip('[')
        img_list[-1] = img_list[-1].rstrip(']')
        img_float = [float(el) for el in img_list]

        # Convert to 1x28x28
        img_tensor = torch.unflatten(torch.tensor(img_float), 0, (1, 28, 28))

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label


def save_confusion_matrix_png(cm, class_labels=None, filename="confusion_matrix.png", title="Confusion Matrix"):
    """
    Save confusion matrix as a PNG file.
    
    Args:
        cm (array-like): Confusion matrix from sklearn.metrics.confusion_matrix
        class_labels (list, optional): List of class labels for the axes
        filename (str): Output filename for the PNG file
        title (str): Title for the confusion matrix plot
    """
    plt.figure(figsize=(10, 8))
    
    if class_labels is None:
        class_labels = range(len(cm))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Confusion matrix saved as '{filename}'.")

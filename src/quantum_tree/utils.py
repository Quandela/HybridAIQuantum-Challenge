from torch.utils.data import Dataset
import os
import pandas as pd
import re
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import seaborn
seaborn.set_theme()

################
## DATA UTILS ##
################

# load the correct train, val dataset for the challenge, from the csv files
class MNIST_partial(Dataset):
    def __init__(self, data = './data', transform=None, split = 'train'):
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
            filename = os.path.join(self.data_dir,'train.csv')
        elif split == 'val':
            filename = os.path.join(self.data_dir,'val.csv')
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
        img_square = torch.unflatten(torch.tensor(img_float),0,(1,28,28))
        if self.transform is not None:
            img_square = self.transform(img_square)
        return img_square, label



####################
## TRAINING UTILS ##
####################

def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    losses = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model.evaluation(images)
            # outputs = nn.functional.log_softmax(outputs, dim=-1)
            losses.append(loss_fn(outputs, labels).item())
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = (correct / total) * 100
    loss = sum(losses) / len(losses)
    return loss, accuracy

def plot_training_metrics_detailed(pickle_files):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    lower_epochs_set = False
    num_epochs = None

    for filename in pickle_files:
        with open(filename, 'rb') as f:
            (train_loss_detailed, train_acc_detailed, train_acc, val_acc, 
             train_loss, val_loss, run_name, _, _) = pickle.load(f)
        
        x_detailed_loss = range(len(train_loss_detailed))
        axes[0, 0].plot(x_detailed_loss, train_loss_detailed, label=run_name)
        
        x_detailed_acc = range(len(train_acc_detailed))
        axes[0, 1].plot(x_detailed_acc, train_acc_detailed, label=run_name)
        
        if not lower_epochs_set:
            num_epochs = len(train_acc)
            lower_epochs_set = True
        epochs = list(range(num_epochs))
        
        axes[1, 0].plot(epochs, train_acc, label=f"{run_name} training")
        axes[1, 0].plot(epochs, val_acc, label=f"{run_name} validation")
        
        axes[1, 1].plot(epochs, train_loss, label=f"{run_name} training")
        axes[1, 1].plot(epochs, val_loss, label=f"{run_name} validation")
    
    axes[0, 0].set_title("Detailed Training Loss")
    axes[0, 0].set_xlabel("Iterations")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    axes[0, 1].set_title("Detailed Training Accuracy")
    axes[0, 1].set_xlabel("Iterations")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    axes[1, 0].set_title("Training and Validation Accuracies")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    axes[1, 1].set_title("Training and Validation Losses")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    
    fig.tight_layout()
    fig.savefig("training_curves.png")

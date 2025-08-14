from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import re
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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

# plot the training curves (accuracy and loss) and save them in 'training_curves.png'
def plot_training_metrics(train_acc,val_acc,train_loss,val_loss):
    fig, axes = plt.subplots(1,2,figsize = (15,5))
    X = [i for i in range(len(train_acc))]
    names = [str(i+1) for i in range(len(train_acc))]
    axes[0].plot(X,train_acc,label = 'training')
    axes[0].plot(X,val_acc,label = 'validation')
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("ACC")
    axes[0].set_title("Training and validation accuracies")
    axes[0].grid(visible = True)
    axes[0].legend()
    axes[1].plot(X,train_loss,label = 'training')
    axes[1].plot(X,val_loss,label = 'validation')
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training and validation losses")
    axes[1].grid(visible = True)
    axes[1].legend()
    axes[0].set_xticks(ticks=X,labels = names)
    axes[1].set_xticks(ticks=X,labels = names)
    fig.savefig("training_curves.png")


# compute the accuracy of the model
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return(torch.tensor(torch.sum(preds == labels).item()/ len(preds)))

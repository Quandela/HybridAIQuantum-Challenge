import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import merlin as ML
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import perceval as pcvl
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from utils import create_circuit, hybridModel, training
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
size_of_dataset = 7000
X, y = mnist['data'][:size_of_dataset], mnist['target'][:size_of_dataset]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, stratify=y)
print('Dataset loaded')

# dimensionality reduction using PCA
d = 8
pca = PCA(n_components=d) 
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print("Dataset's dimension reduced")

# scaling the data for the quantum layer (must be in [0,1])
scaler = MinMaxScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
y_train, y_test = torch.tensor(y_train.cat.codes.values, dtype=torch.long), torch.tensor(y_test.cat.codes.values, dtype=torch.long)
print('Dataset scaled')

# defining the photonic circuit
m = d
c = create_circuit(m=m, d=d, depth=2)

# instantiate the hybrid model
hybrid_model = hybridModel(m, d)

# training the model

# hyperparameters
n_epochs = 10
batch_size = 64
lr = 0.01
m = 8
d = 8

# dataloaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# tracking accuracies
train_accs, test_accs = training(hybrid_model, train_loader, test_loader, n_epochs, batch_size, lr, X_train, y_train, verbose=True)

# plot accuracies over time
plt.figure(figsize=(8, 6))
plt.plot(train_accs, label='Train accuracy')
plt.plot(test_accs, label='Test accuracy')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model Accuracies over 10 Epochs")
plt.show()
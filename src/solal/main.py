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
from merlin.datasets import mnist_digits

def load_data():
    # Load the MNIST dataset
    print("\n Loading MNIST dataset from MerLin split... ")
    X_train, y_train, metadata = mnist_digits.get_data_train_percevalquest()
    X_test, y_test, metadata = mnist_digits.get_data_test_percevalquest()
    print(f'Dataset loaded with X_train of shape {X_train.shape} and y_train of shape {y_train.shape}')

    print("[Data procesing]: data is flattened")
    # Flatten the images before PCA
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # dimensionality reduction using PCA
    d = 8
    print(f"[Data procesing]: PCA is applied with {d} components")
    pca = PCA(n_components=d)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # scaling the data for the quantum layer (must be in [0,1])
    print("[Data procesing]: MinMax Scaling is applied")
    scaler = MinMaxScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)
    print("[Data procesing]: data is scaled and ready for training \n")

    return X_train, y_train, X_test, y_test, d

def plot_training_curves(train_accs, test_accs):
    # plot accuracies over time
    plt.figure(figsize=(8, 6))
    plt.plot(train_accs, label='Train accuracy')
    plt.plot(test_accs, label='Test accuracy')

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Model Accuracies over 10 Epochs")
    plt.savefig("Training_accuracies.png")
    print("\n Training curves saved to /Training_accuracies.png !")
    plt.show()


def train_model(X_train, y_train, X_test, y_test, d):
    # defining the photonic circuit
    m = d
    print("[Quantum circuit]: creating circuit with perceval")
    c = create_circuit(m=m, d=d, depth=2)

    # instantiate the hybrid model
    print("[Quantum circuit]: creating Hybrid model with MerLin")
    hybrid_model = hybridModel(m, d)

    # training the model
    # hyperparameters
    n_epochs = 15
    batch_size = 64
    lr = 0.01
    print(f"Training settings: \n - n_epochs: {n_epochs}, batch_size: {batch_size}, lr: {lr}")

    # dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # tracking accuracies
    print("[Hybrid model training]: the hybrid model is trained")
    train_accs, test_accs = training(hybrid_model, train_loader, test_loader, n_epochs, batch_size, lr, X_train, y_train, verbose=True)

    return train_accs, test_accs


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, d = load_data()
    train_accs, test_accs = train_model(X_train, y_train, X_test, y_test, d=d)
    plot_training_curves(train_accs, test_accs)


#!/usr/bin/env python3
"""
Main script to train and evaluate quantum hybrid neural network models
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from merlin.datasets import mnist_digits
import time

from utils import (
    get_device,
    evaluate_model,
    plot_training_metrics
)
from models import (
    Type2_TrainableKernel__Hybrid,
    Type2_TrainableKernel__Hybrid__2_layers,
    Type2_TrainableKernel__Hybrid__2_layers__combined_parallel,
    LightweightCNN
)


def train_model(model, train_dataloader, criterion, optimizer, device, num_epochs=10, save_embeddings=False, save_dir="./logs"):
    """Train the quantum hybrid neural network model.
    
    Args:
        model: The quantum hybrid neural network model to train
        train_dataloader: DataLoader containing training data
        criterion: Loss function (e.g., CrossEntropyLoss)
        optimizer: Optimizer (e.g., Adam)
        device: Device to run training on (CPU/GPU)
        num_epochs: Number of training epochs (default: 10)
        save_embeddings: Whether to save quantum layer embeddings (default: False)
        save_dir: Directory to save embeddings and logs (default: "./logs")
    
    Returns:
        tuple: (accuracies_list, losses_list) for each epoch
    """
    # Track training metrics for plotting
    accuracies = []
    losses = []
    
    # Create directory structure for saving embeddings if requested
    if save_embeddings:
        os.makedirs(save_dir, exist_ok=True)
    
    # Main training loop
    for epoch in range(num_epochs):
        # Create epoch-specific directories for embedding storage
        if save_embeddings:
            epoch_embeddings_dir_l1 = os.path.join(save_dir, f"epoch_{epoch}", "layer_1")
            epoch_embeddings_dir_l2 = os.path.join(save_dir, f"epoch_{epoch}", "layer_2")
            os.makedirs(epoch_embeddings_dir_l1, exist_ok=True)
            os.makedirs(epoch_embeddings_dir_l2, exist_ok=True)

        # Set model to training mode
        model.train()
        # Initialize epoch metrics
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Create progress bar for batch processing
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move data to the appropriate device (CPU/GPU)
            images, labels = images.to(device), labels.to(device)

            # Reset gradients from previous batch
            optimizer.zero_grad()
            
            # Forward pass - handle models that return embeddings vs. those that don't
            if len(model(images)) == 3:  # Model returns (outputs, conv1_embeddings, conv2_embeddings)
                outputs, conv1_embeds, conv2_embeds = model(images)
                
                # Save quantum layer embeddings for analysis (only for specific batch to save space)
                if save_embeddings and batch_idx == 1:  # Save embeddings for batch 1
                    np.save(os.path.join(epoch_embeddings_dir_l1, f"batch_{batch_idx:05d}.npy"), 
                           conv1_embeds.cpu().numpy())
                    np.save(os.path.join(epoch_embeddings_dir_l2, f"batch_{batch_idx:05d}.npy"), 
                           conv2_embeds.cpu().numpy())
            else:
                # Simple model that only returns outputs
                outputs = model(images)
            
            # Calculate loss and perform backpropagation
            loss = criterion(outputs, labels)
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters

            # Accumulate batch loss (weighted by batch size)
            running_loss += loss.item() * images.size(0)
            
            # Calculate batch accuracy
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class indices
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar with current metrics
            current_acc = 100 * correct_predictions / total_predictions
            current_loss = running_loss / (batch_idx + 1) / images.size(0)
            pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc:.2f}%'})

        # Calculate final epoch metrics
        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_accuracy = 100 * correct_predictions / total_predictions

        print(f"Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # Store metrics for plotting
        accuracies.append(epoch_accuracy)
        losses.append(epoch_loss)
    
    return accuracies, losses


def main():
    """Main function to train and evaluate quantum hybrid neural networks.
    
    Supports four model architectures:
    - single: Single quantum layer hybrid model
    - 2layer: Two quantum layer hybrid model  
    - combined: Parallel classical and quantum branches combined model
    - cnn: Lightweight classical CNN for comparison
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Train Quantum Hybrid Neural Networks')
    parser.add_argument('--model', type=str, choices=['single', '2layer', 'combined', 'cnn'], 
                       default='single', help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='./data', 
                       help='Directory containing MNIST .npy files')
    parser.add_argument('--save-model', type=str, default=None, 
                       help='Path to save trained model')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to load pre-trained model')
    parser.add_argument('--save-embeddings', action='store_true',
                       help='Save embeddings during training')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory to save logs and embeddings')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of workers for data loading')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate the model (requires --load-model)')
    
    args = parser.parse_args()
    
    # Determine the best available computing device (CPU/GPU/MPS)
    device = get_device()
    print(f"Using device: {device}")
    
    # Load MNIST dataset using Perceval Quest format
    print("\n Loading MNIST data...")
    X_train, y_train, metadata = mnist_digits.get_data_train_percevalquest()
    X_test, y_test, metadata = mnist_digits.get_data_test_percevalquest()

    # Convert numpy arrays to PyTorch tensors and add channel dimension for CNN compatibility
    # Transform grayscale images from [N, H, W] to [N, 1, H, W] format
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
    y_test_tensor = torch.LongTensor(y_test)

    # Create PyTorch TensorDatasets for efficient data loading
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders for batch processing during training and evaluation
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Shuffle training data for better learning
        num_workers=args.num_workers  # Parallel data loading workers
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=args.num_workers
    )
    print(f"... Data loaded !")
    
    # Initialize the selected quantum hybrid model architecture
    print(f"Initializing {args.model} model...")
    if args.model == 'single':
        # Single quantum convolutional layer model
        model = Type2_TrainableKernel__Hybrid(num_classes=10, device=device)
    elif args.model == '2layer':
        # Two quantum convolutional layers model
        model = Type2_TrainableKernel__Hybrid__2_layers(num_classes=10, device=device)
    elif args.model == 'combined':
        # Combined classical and quantum parallel branches model
        model = Type2_TrainableKernel__Hybrid__2_layers__combined_parallel(num_classes=10, device=device)
        args.lr = 1e-5  # Use lower learning rate for the more complex combined model
    elif args.model == 'cnn':
        # Lightweight classical CNN for comparison
        model = LightweightCNN(num_classes=10)
    
    # Move model to the selected device and display parameter count
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Load pre-trained model weights if provided
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        model.load_state_dict(torch.load(args.load_model, map_location=device))
    
    # Setup loss function and optimizer for training
    criterion = nn.CrossEntropyLoss().to(device)  # Standard loss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # Adam optimizer with specified learning rate
    
    # Training phase (skip if eval_only is specified)
    if not args.eval_only:
        # Train the model for the specified number of epochs
        print(f"Starting training for {args.epochs} epochs...")
        accuracies, losses = train_model(
            model, train_dataloader, criterion, optimizer, device,
            num_epochs=args.epochs, save_embeddings=args.save_embeddings,
            save_dir=args.log_dir
        )
        
        # Visualize training progress with accuracy and loss plots
        plot_training_metrics(accuracies, losses, f"{args.model.title()} Model Training")
        
        # Save trained model weights if path is specified
        if args.save_model:
            torch.save(model.state_dict(), args.save_model)
            print(f"Model saved to {args.save_model}")
    
    # Final evaluation on the test set
    print("Evaluating on test set...")
    start_time = time.time()
    test_loss, test_accuracy = evaluate_model(model, test_dataloader, criterion, device)
    eval_time = time.time() - start_time
    
    # Display comprehensive results
    print(f"\n=== Model Performance Summary ===")
    print(f"Model type: {args.model.upper()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Evaluation Time: {eval_time:.2f} seconds")
    
    # Provide context comparison for CNN model
    if args.model == 'cnn':
        print(f"\n--- Classical CNN Baseline ---")
        print("This lightweight CNN serves as a baseline for comparison with quantum hybrid models.")
        print("Key characteristics:")
        print("- Pure classical convolution operations")
        print("- Standard CNN architecture patterns")


if __name__ == "__main__":
    main()
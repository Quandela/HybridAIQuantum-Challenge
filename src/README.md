# HybridAIQuantum Challenge

### A Transfer Learning approach


A transfer learning implementation that combines classical ResNet18 architecture with quantum computing layers using Perceval and MerLin. The script provides both quantum and classical classification modes for MNIST digit recognition:

- **Quantum Mode**: Integrates quantum photonic circuits as the final classification layer, replacing traditional fully connected layers with quantum interference patterns
- **Classical Mode**: Uses standard neural network layers for comparison
- **Transfer Learning**: Leverages pre-trained ResNet18 weights from ImageNet, freezing the backbone and training only the classification head
- **Flexible Configuration**: Supports custom digit selection, quantum circuit parameters, and training hyperparameters

![Transfer Learning Model Architecture](QuantumNomad/TL_model.png)

**Key Features:**
- Quantum circuit construction with configurable modes and encoding schemes
- Hybrid classical-quantum neural network training
- MNIST dataset preprocessing and augmentation
- Training metrics visualization and model evaluation
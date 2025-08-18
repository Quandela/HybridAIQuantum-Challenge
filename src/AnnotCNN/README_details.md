# Hybrid Quantum-Classical MNIST Pipeline  
*(by the Perceval Quantum2Pi Knights)*

This repository implements a hybrid machine learning pipeline that combines a classical Convolutional Neural Network (CNN) with a quantum embedding layer based on photonic boson sampling using Quandela’s Perceval library. Developed by the Quantum2Pi Knights, the project explores how quantum-enhanced feature encoding can improve MNIST digit classification while allowing easy switching between local simulation and remote quantum execution (e.g. via Scaleway QaaS).


## Key Features

- **Hybrid Model Architecture:**  
  - A CNN-based feature extractor transforms 28×28 images into a compact representation.
  - A quantum embedding layer (via a BosonSampler) maps the CNN features into a quantum circuit, enabling interference-based encoding.
  
- **Modular Quantum Backend:**  
  - Seamlessly switch between local simulation and remote execution on Scaleway QaaS via external configuration.
  
- **Interchangeable Optimization Methods:**  
  - Supports multiple hyperparameter search strategies (e.g., TPE, grid search, Optuna) via a unified interface.
  
- **Performance Enhancements:**  
  - **Caching:** Avoid redundant quantum computations with in‑memory and optional disk‑based persistent caching.  
  - **Parallel Processing:** Utilize multi-threading or multiprocessing (via `concurrent.futures` or `torch.multiprocessing`) to compute quantum embeddings in parallel.  
  - **Sparse Input Masking:** Optionally drop or mask portions of input images (structured or random) to reduce computational load and act as a form of data augmentation.  
  - **Vectorized Operations:** Leverage PyTorch/NumPy vectorization to speed up mathematical operations and data processing.

- **Configurable via YAML:**  
  - All critical parameters (such as quantum modes, optimization strategy, caching options, and backend selection) are set via external YAML config files.


**Folder Overview:**


- **`src/`**  
  Folder contains all the source code for the classical and quantum models.
  Contains all core modules:  
  - *boson_sampler.py*: Implements the quantum embedding using Perceval. 
  - *utils.py*: Mixed utilities. Loads and preprocesses MNIST data (with optional pixel masking).  Configs to switch between different setups (e.g., local vs. remote quantum backends, various hyperparameter optimizers), to use with the yaml file. Data utilities, contains helper functions such as accuracy and plotting,for plotting and embedding utils.
  - *model_quantum.py*: Defines the hybrid CNN+quantum model.  
  - *optimization.py*: Provides a unified interface for various hyperparameter search methods (TPE, grid search, Optuna, etc.).  
  - *quantum_backend.py*: Manages the selection between local simulation and remote QaaS (e.g., Scaleway).  
  - *main.py*: Experiments scripts for running full training or optimization experiments, with logs and checkpoints stored in the `results/` subfolder.



- **`data/`**  
  - Holds the dataset of the MNIST files. `train.csv` and `test.csv` files (the reduced MNIST dataset).
  - `pca_model.pkl`: The PCA model used to reduce the dimensionality of the images. This model is automatically generated during training and can be reused in subsequent experiments (including the quantum embedding pipeline).


- **`results/`**  
  Where outputs are saved.  Stores all the outputs, including visualizations and performance reports.

- **`examples/`**  
  All the pdf document/papers related to the [First Perceval Quest](https://perceval.quandela.net/forum/t/first-perceval-quest-last-call-for-registration/217)



---



## Setup

This project uses [Conda](https://docs.conda.io/en/latest/) for environment management. To set up the environment, run the following commands:

```bash
cd project-name
conda env create -f environment.yml
conda activate q2pi-hybrid-quantum-env

#in case of errors:
#conda deactivate
#conda env remove --name q2pi-hybrid-quantum-env
#conda env list
```








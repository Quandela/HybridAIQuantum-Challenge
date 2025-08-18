## Overview
This project explored a hybrid photonic-classical MNIST classification that prioritizes methodological originality over accuracy within the Perceval Quest constraints (6k-image subset). We combine a lightweight classical classifier with a photonic feature extractor using a Perceval circuit with trainable interferometric blocks and input-dependent phase encoders. The direct-circuit formulation uses classical features to modulate phase shifters for data encoding.

We use bilinear or PCA-based compression to downsample images, map phase parameters to drive a boson-sampling circuit using GenericInterferometer meshes (triangular/rectangular variants) with optional post-selection, and create a quantum embedding vector from detection outcomes to concatenate with the classifier's input. A classical PCA baseline with the same parameter budget for unbiased comparison, training/evaluation utilities, and hooks for remote simulation via Scaleway's QaaS (used in the Quest) are also in the codebase. 


Initial tests indicate that photonic embedding may operate as a feature map with low parameter counts. However, due to limited dataset size and simulator/QPU constraints, accuracy deltas should not be overinterpreted. Instead, we prioritize reproducibility (fixed configs, logged seeds, explicit downsampling), ablation levers (encoding, interferometer shape, sampling strategy), and MerLin's QuantumLayer future compatibility for benchmarking and speedups.

![Alt text](./overview_chart.png)

## Setup
This project uses [Conda](https://docs.conda.io/en/latest/) for environment management. To set up the environment, run the following commands:

```bash
cd project-name
conda env create -f environment.yml
conda activate q2pi-hybrid-quantum-env
```


## Run
This project can be run in 2 modes:

**Classical mode:**

`python3 src/main.py classic`

**Quantum hybrid mode:**

`python3 src/main.py hybrid`

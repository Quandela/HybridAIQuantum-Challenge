# Lancelot LOQC MNIST

This experiment was developed for the Perceval Quest, a joint Quandela/Scaleway challenge. The repository explores how Linear Optical Quantum Computing (LOQC) can augment a classical convolutional neural network (CNN) on the MNIST handwritten digits task.

## Project Structure

- `main.py` — unified training entry-point for both the classical CNN and the hybrid quantum-classical model.
- `modelparams/` — default directory where trained weights, confusion matrices, and plots are stored.
- `draw_digits.py`, `label_distribution.py`, etc. — helper scripts used during exploration.
- Legacy notebooks (`classic_conv_model.ipynb`, `hybrid_conv_model.ipynb`) are kept for reference; their logic now lives in `main.py`.

By default the script pulls the Perceval Quest MNIST splits directly from the MerLin datasets package. If you prefer supplying your own CSV files (`train.csv`, `val.csv`), store them in a directory such as `src/AnnotCNN/data` and launch the CLI with `--data-source csv --data-dir <path>`.

## Classical Baseline

The classical model builds a lightweight CNN over 28×28 grayscale inputs:

1. Two convolution + max-pooling blocks.
2. Flattening followed by a dense stack with dropout.
3. Softmax output layer over the 10 MNIST classes.

A smaller "light" variant is also provided for quick experimentation on the pooled 14×14 resolution. Training uses categorical cross-entropy with Adam and produces accuracy/loss curves and a confusion matrix.

## Hybrid LOQC Model

The hybrid approach maps each pooled image to an interferometer built with Perceval components:

1. Images are down-sampled via max pooling, reshaped into Hamiltonians, and converted to unitary matrices using a dilation trick.
2. A brickwork circuit composed of programmable beam splitters and generic two-mode gates encodes optical parameters (`BS_params`, `omega_params`).
3. Photon sampling (with optional Scaleway remote acceleration) estimates mean spatial distributions which serve as features for a classical dense classifier.
4. Optical parameters are optimised with SPSA or CMA-ES, while the classical layer is fine-tuned with gradient descent.

Artifacts such as learned optical parameters and confusion matrices are saved under the configured output directory.

## Environment Setup

```bash
cd src/Lancelot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Hybrid training additionally needs:

- `perceval` (included in `requirements.txt`)
- `cma` (for CMA-ES optimisation)
- `merlin-datasets` (provides the Perceval Quest MNIST splits used by default)
- Optional Scaleway credentials (`SCW_PROJECT_ID`, `SCW_SECRET_KEY`) for GPU-backed simulations. Without them the processor falls back to the local simulator.

## Command Line Usage

All training is orchestrated through `main.py`:

```bash
# Classical CNN (full variant using MerLin splits)
python main.py classical \
  --output-dir modelparams/classical_run \
  --epochs 30 \
  --batch-size 64

# Classical CNN (light variant, skip plots, custom CSV path)
python main.py classical \
  --data-source csv \
  --data-dir ../../AnnotCNN/data \
  --variant light \
  --skip-plots

# Hybrid LOQC model with CMA-ES (2 quantum epochs)
python main.py hybrid \
  --output-dir modelparams/hybrid_run \
  --optimizer cmaes \
  --quantum-epochs 2 \
  --num-samples 500
```

Useful switches:

- `--data-source {merlin,csv}` toggles between the bundled MerLin splits and custom CSV files.
- `--pool-size` controls the initial max pooling factor (default 2).
- `--hidden-units`, `--sigma`, `--num-layers`, `--num-samples` tune the hybrid optimiser.
- `--log-level DEBUG` increases verbosity for troubleshooting.

## Outputs

- Classical mode saves a compiled Keras model (`classical_<variant>_model.keras`), training history plots, and confusion matrices (both `.png` and `.npy`).
- Hybrid mode saves the optical parameters (`BS_params.npy`, `omega_params.npy`), loss/accuracy traces, and a confusion matrix plot.

Refer to `mean_output_vs_samples.png` for convergence analysis of the photon sampling procedure, and to the archived notebooks for derivations and exploratory analysis.

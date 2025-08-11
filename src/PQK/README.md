# Quantum Hybrid Neural Networks (PQK)

This directory contains scripts to train and evaluate quantum hybrid neural networks based on the examples notebook.

## Files

- `main.py` - Main training script
- `models.py` - Model definitions (single layer, 2-layer, and combined models)
- `utils.py` - Utility functions for data loading, quantum circuits, and evaluation
- `requirements.txt` - Python dependencies

## Usage

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train models

Single quantum layer model:
```bash
python main.py --model single --epochs 10
```

Two quantum layer model:
```bash
python main.py --model 2layer --epochs 10
```

Combined classical-quantum model:
```bash
python main.py --model combined --epochs 10
```

### Additional options

- `--save-model MODEL_PATH` - Save trained model
- `--load-model MODEL_PATH` - Load pre-trained model
- `--save-embeddings` - Save quantum embeddings during training
- `--eval-only` - Only evaluate (requires --load-model)
- `--batch-size BATCH_SIZE` - Training batch size (default: 32)
- `--lr LEARNING_RATE` - Learning rate (default: 1e-3)

### Example with all options
```bash
python main.py --model combined --epochs 20 --batch-size 64 --lr 1e-5 \
                --save-model ./trained_model.pth \
                --save-embeddings --log-dir ./training_logs
```

## Models

1. **Single Layer Model** (`Type2_TrainableKernel__Hybrid`)
   - One quantum convolutional layer
   - Classical fully connected layers

2. **Two Layer Model** (`Type2_TrainableKernel__Hybrid__2_layers`)
   - Two quantum convolutional layers
   - Classical fully connected layers

3. **Combined Model** (`Type2_TrainableKernel__Hybrid__2_layers__combined_parallel`)
   - Parallel classical and quantum branches
   - Feature fusion layer
   - Best performance from the notebook examples
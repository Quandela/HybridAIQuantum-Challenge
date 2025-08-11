## Quantum Hybrid Neural Networks (PQK)

## Models

1. **Single Layer Model** (`Type2_TrainableKernel__Hybrid`)
   - One quantum convolutional layer
   - Classical fully connected layers

```bash
python main.py --model single --epochs 10 --data-dir ./data
```

2. **Two Layer Model** (`Type2_TrainableKernel__Hybrid__2_layers`)
   - Two quantum convolutional layers
   - Classical fully connected layers

Two quantum layer model:
```bash
python main.py --model 2layer --epochs 10 --data-dir ./data
```

3. **Combined Model** (`Type2_TrainableKernel__Hybrid__2_layers__combined_parallel`)
   - Parallel classical and quantum branches
   - Feature fusion layer
   - Best performance from the notebook examples

```bash
python main.py --model combined --epochs 10 --data-dir ./data
```

### Example with all options
```bash
python main.py --model combined --epochs 20 --batch-size 64 --lr 1e-5 \
                --data-dir ./data --save-model ./trained_model.pth \
                --save-embeddings --log-dir ./training_logs
```


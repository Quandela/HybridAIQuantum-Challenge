# Models proposed

## A Photonic Quantum Neural Network
The photonic quantum Neural Network is a hybrid quantum classical model made of trainable generic interferometers (in purple), an encoding layer (in gray), a learnable scale layer and a linear layer for class mapping.
<div align="center">
  <img width="54%" alt="Challenge-img" src="./photonic_qNN/photonic_qNN.png">
</div>

## GLASE
GLASE is a QNN framework that estimates the gradients of the photonic QNNs using a seperate classical surrogate model during training to avoid training instabilities or barren plateaus.
### Training the Model

Run the training script with default hyperparameters or specify your own using command-line arguments. For example:

```bash
python src/quantum_tree/train.py --m 20 --n 3 --batch_size 256 --lr 2e-3 --weight_decay 1e-3 --epochs 50 --label_smoothing 0.1
```

### Plotting Training Results

To visualize the training metrics after training both QNN and CNN, use the plotting script.

```bash
python src/quantum_tree/plot_result.py --files qnn.pkl cnn.pkl
```

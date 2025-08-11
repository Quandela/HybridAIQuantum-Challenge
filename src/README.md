# Models proposed

## A Photonic Quantum Kernel Support Vector Machine

A quantum kernel specifically adapted to photonic quantum computers was developed and compared against classical support vector machines (SVMs). The quantum model uses a photonic interferometer to encode PCA-processed images into a high-dimensional quantum Hilbert space. Photon measurements yield a fidelity-based kernel, which is then passed to a classical SVM for classification.

## A Photonic QNN as a features extractor

A hybrid classical quantum photonic neural network is implemented and trained on a subset of MNIST. 
After reducing the images sizes using PCA, the quantum photonic circuit is used as feature extractor. 
The quantum layer is implemented using the `merlin` library.
These quantum features are then concatenated with the classical ones to classify the input images.
The penultimate classical layer is used to shift the classes's disttibution closer to the uniform one.

Besides MNIST, the quantum layer is also shown to work with some classical non linearly separable dataset such as the moons and circles (cf `examples/pqnn.ipynb`).


### To train and test the model

```bash
python solal/main.py
```

## A Photonic Quantum Neural Network
The photonic quantum Neural Network is a hybrid quantum classical model made of trainable generic interferometers (in purple), an encoding layer (in gray), a learnable scale layer and a linear layer for class mapping.
<div align="center">
  <img width="54%" alt="Challenge-img" src="./photonic_qNN/photonic_qNN.png">
</div>

## Tristan : A quantum 2d convolution-based network implementation
Inspired by [S. Shi, et al](https://arxiv.org/pdf/2303.03707), our implementation (Fig. 1) replaces the dot product of traditional convolution by a quantum circuit while keeping the sliding window principle. Our quantum implementation is a photonic circuit with two parts: a feature map and an ansatz. The feature map uses a fixed input Fock state and contains beam splitters and phase shifters, with some parameters that are fixed and others that depend on input pixel values.

- [See our topology](tristan/qml/models/topologies/tristan.py)
- [See our feature map](tristan/qml/models/layers/quantum/feature_maps/achilles.py)
- [See our ansatz](tristan/qml/models/layers/quantum/ansatz/penarddun.py)
- [See our paper](tristan/docs/qaradoq-report.pdf)


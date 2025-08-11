# A Photonic QNN as a features extractor

A hybrid classical quantum photonic neural network is implemented and trained on a subset of MNIST. 
After reducing the images sizes using PCA, the quantum photonic circuit is used as feature extractor. 
The quantum layer is implemented using the `merlin` library.
These quantum features are then concatenated with the classical ones to classify the input images.
The penultimate classical layer is used to shift the classes's disttibution closer to the uniform one.

Besides MNIST, the quantum layer is also shown to work with some classical non linearly separable dataset such as the moons and circles (cf `examples/pqnn.ipynb`).


# To train and test the model

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/main.py
```


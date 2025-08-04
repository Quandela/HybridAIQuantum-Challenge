# The First Perceval Quest - Qaradoq Team

## IMPORTANT NOTES!

### To run it
The codebase was made during the First Perceval Quest by [Quandela](https://quandela.com) and [Scaleway](https://scaleway.com).
Few modifications were done to make it run 'as it':

```bash
python3 main.py # should be enough to train the Tristan model
```

### MerLin update
This small QML framework is not -yet- based on [MerLin](https://merlinquantum.ai/) from Quandela.

Effort must be made on the [qconv2d layer](./qml/models/layers/quantum/qconv_2d.py) which is based on the [SLOS directory](./qml/models/layers/quantum/slos) (which must be replaced by MerLin).

## Getting Started

### Setup
To install the python environment:
```bash
make install
```

### Training
==> You need to update [main.py](main.py) first

To train our quantum ML model baseline 'Tristan' (composed of the 'Achilles' feature map and the 'Pernarddun' ansatz):
```bash
make mnist-tristan
```

You can also train 'Dagonet' (composed of the 'Odysseus' feature map and the 'Gofanon' ansatz):
```bash
make mnist-dagonet
```

**About output interpretation**

The default output interpretation method is "argmax" (select the most probable output state vector).
To use the "cumulative" method (sum of probabilities), edit `qml/models/layers/quantum/qconv2d.py`:
- Comment the lines tagged "Output - ARGMAX"
- Uncomment the lines tagged "Output - CUMULATIVE"


### Inference
==> You need to update [main.py](main.py) first

To put an trained model in inference mode:
```bash
python3 main.py predict --model-directory path/to/model/dir/

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃      Validate metric      ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│       val_accuracy        │    0.9383333444595337     │
└───────────────────────────┴───────────────────────────┘
```

Add option `--confusion_matrix_out` to compute the confusion matrix and save it to a file.
The output format is inferred from the file extension. Supported formats include PNG, JPEG, SVG and PDF.
```bash
python3 main.py predict --model-directory path/to/model/dir/ --confusion_matrix_out="path/to/confusion_matrix.png"
```

## Project Architecture

### [Data](./qml/data/)
Raw data for MNIST, dataset object and dataload for training time

### [Models](./qml/models/)
Model for training and inference. Concerns everything related to topologies and layers (standards, custom, classical or quantum-based). The main purpose is to keep a maximum re-usability.

Model topology is defined in to [topologies](./qml/models/topologies/) folder, see [examples](./qml/models/topologies/examples.py).
New layer can be defined in the [layers](./qml/models/layers/) folder.

By using properly this framework to define your `wonderbar_topology`, you will be able to use:
```bash
python3 main.py train --topology wonderbar_topology
```

### [Training](./qml/training/)
Everything related to training hyperparameters such as optimizer, objective function (loss) and learning-rate scheduler. Not used during inference.

### [Inference](./qml/inference/)
Some tools to evaluate an existing / already trained model.


## Team Members
- Antoine Radet (aradet@scaleway.com)
- Ludovic Le Frioux (llefrioux@scaleway.com)
- Valentin Macheret (vmacheret@scaleway.com)
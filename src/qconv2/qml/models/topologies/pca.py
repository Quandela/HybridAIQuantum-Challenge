import torch
import qml.models.layers as layers

from typing import List

from qml.models.topologies.register import register
from qml.models.topologies.sequence import topology


# This topology must be done with a full dataset size
@register("pca")
def pca_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]
    components = extra_params["components"]

    return [
        topology(
            path=f"output/layers/pca_{components}",
            sequence=[
                layers.flatten(),
                layers.pca(components),
            ],
        ),
        layers.linear(output_size=nb_output_classes),
    ]

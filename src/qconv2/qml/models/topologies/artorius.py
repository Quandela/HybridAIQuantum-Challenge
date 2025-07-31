import torch
import qml.models.layers as layers

from typing import List

from qml.models.topologies.register import register
from qml.models.topologies.sequence import topology


@register("artorius")
def artorius_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]
    components = extra_params.get("components", 10)

    return [
        topology(f"output/layers/pca_{components}", learnable=False),
        layers.concat(
            sequences=[
                [
                    layers.identity(),
                ],
                [
                    layers.identity(),
                ],
            ]
        ),
        layers.linear(output_size=nb_output_classes),
    ]


@register("artorius_qt")
def artorius_qt_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]
    components = extra_params.get("components", 10)

    return [
        topology(f"output/layers/pca_{components}", learnable=False),
        layers.concat(
            sequences=[
                [
                    layers.slos_circuit(
                        fmap="odysseus",
                        ansatz="gofanon",
                        output_size=components,
                    ),
                ],
                [
                    layers.identity(),
                ],
            ]
        ),
        layers.linear(output_size=nb_output_classes),
    ]

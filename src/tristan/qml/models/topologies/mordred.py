import torch
import qml.models.layers as layers

from typing import List

from qml.models.topologies.register import register
from qml.models.topologies.sequence import topology


@register("mordred")
def mordred_topology(
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


@register("mordred_qt")
def mordred_qt_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]
    components = extra_params.get("components", 10)

    return [
        topology(f"output/layers/pca_{components}", learnable=False),
        layers.concat(
            sequences=[
                [
                    layers.photonic_circuit(
                        fmap="hector",
                        ansatz="gofanon",
                        gradient_method="spsa",
                        converter="cumulative_mode",
                        thresholded_output=False,
                    ),
                ],
                [
                    layers.identity(),
                ],
            ]
        ),
        layers.linear(output_size=nb_output_classes),
    ]

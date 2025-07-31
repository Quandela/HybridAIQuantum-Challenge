import torch
import qml.models.layers as layers

from typing import List

from qml.models.topologies.register import register
from qml.models.topologies.sequence import topology


@register("bohort")
def bohort_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]
    components = extra_params.get("components", 10)

    return [
        topology(f"output/layers/pca_{components}", learnable=False),
        layers.linear(output_size=nb_output_classes),
    ]


@register("bohort_qt")
def bohort_qt_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]
    components = extra_params.get("components", 10)

    return [
        layers.concat(
            sequences=[
                [
                    topology(f"output/layers/pca_{components}", learnable=False),
                    layers.photonic_circuit(
                        fmap="teucros",
                        iterations=1,  # whole circuit = iterations x (fmap + ansatz)
                        gradient_method="dummy",  # Can be updated by 'spsa' but is twice slower
                        converter="cumulative_mode",  # Cumulative probability of having of photons in a particular mode
                        thresholded_output=False,
                    ),
                ],
                [
                    topology(f"output/layers/pca_{components}", learnable=False),
                ],
            ],
        ),
        layers.linear(output_size=nb_output_classes),
    ]

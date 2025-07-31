import torch
import qml.models.layers as layers

from typing import List

from qml.models.topologies.register import register
from qml.models.topologies.sequence import topology


@register("galahad")
def galahad_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]
    components = extra_params.get("components", 10)

    return [
        layers.concat(
            sequences=[
                [
                    topology(f"output/layers/pca_{components}", learnable=False),
                    layers.fourier_features_1d(components * 2, 0.1),
                ],
                [
                    topology(f"output/layers/pca_{components}", learnable=False),
                ],
            ]
        ),
        layers.mish(),
        layers.linear(output_size=nb_output_classes),
    ]


@register("galahad_qt")
def galahad_qt_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]
    components = extra_params.get("components", 10)

    return [
        layers.concat(
            sequences=[
                [
                    topology(f"output/layers/pca_{components}", learnable=False),
                    layers.fourier_features_1d(components * 2, 0.1),
                    layers.photonic_circuit(
                        fmap="ajax",
                        converter="cumulative_mode",
                        thresholded_output=False,
                    ),
                ],
                [
                    topology(f"output/layers/pca_{components}", learnable=False),
                ],
            ]
        ),
        layers.mish(),
        layers.linear(output_size=nb_output_classes),
    ]

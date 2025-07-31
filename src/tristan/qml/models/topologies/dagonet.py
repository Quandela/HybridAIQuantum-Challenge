import torch
import qml.models.layers as layers

from typing import List

from qml.models.topologies.register import register


@register("dagonet")
def dagonet_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]

    return [
        layers.conv_2d(kernel=3, stride=2, output_chan=9, bias=False),
        layers.flatten(),
        layers.linear(output_size=nb_output_classes),
    ]


@register("dagonet_qt")
def dagonet_qt_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]

    return [
        # QConv2D_AdvancedFeatureMap
        layers.qconv_2d(
            fmap="odysseus",
            ansatz="gofanon",
            kernel=3,
            stride=2,
            advanced_input_mapping=True,
        ),
        layers.flatten(),
        layers.linear(output_size=nb_output_classes),
    ]

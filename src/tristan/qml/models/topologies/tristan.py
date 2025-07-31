import torch
import qml.models.layers as layers

from typing import List

from qml.models.topologies.register import register


@register("tristan")
def tristan_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]

    return [
        layers.conv_2d(kernel=3, stride=2, output_chan=9, bias=False),
        layers.flatten(),
        layers.linear(output_size=nb_output_classes),
    ]


@register("tristan_qt")
def tristan_qt_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]

    return [
        # QConv2D_SimplePhaseEncoding
        layers.qconv_2d(fmap="achilles", ansatz="penarddun", kernel=3, stride=2),
        layers.flatten(),
        layers.linear(output_size=nb_output_classes),
    ]

import torch
import qml.models.layers as layers

from typing import List

from qml.models.topologies.register import register
from qml.models.topologies.sequence import topology


DUMMY_LINEAR_TOPOLOGY = "dummy_linear"


@register("save")
def dummy_save_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]

    return [
        layers.flatten(),
        topology(
            path="output/layers/dummy_1",
            sequence=[
                layers.linear(150),
                layers.prelu(),
                layers.linear(50),
                layers.prelu(),
            ],
        ),
        layers.linear(output_size=nb_output_classes),
    ]


@register("load")
def dummy_load_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]

    return [
        layers.flatten(),
        topology("output/layers/dummy_1", learnable=False),
        layers.linear(25),
        layers.relu(),
        layers.linear(output_size=nb_output_classes),
    ]


@register(DUMMY_LINEAR_TOPOLOGY)
def dummy_linear_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]

    return [
        layers.flatten(),
        layers.linear(output_size=nb_output_classes),
    ]


@register("dummy_hybrid")
def dummy_hybrid_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]

    return [
        layers.flatten(),
        layers.boson_sampling(modes=5, photons=2),
        layers.linear(output_size=nb_output_classes),
    ]


@register("dummy_conv_2d")
def dummy_conv_2d_topology(
    input_shape: torch.Size, output_shape: torch.Size, **extra_params
) -> List:
    nb_output_classes = output_shape[0]

    return [
        layers.conv_2d(kernel=3, stride=2, output_chan=9, bias=False),
        layers.flatten(),
        layers.linear(output_size=nb_output_classes),
    ]


@register("dummy_fc")
def dummy_fc_topology(
    input_shape: torch.Size, nb_output_classes: int, **extra_params
) -> List:
    return [
        layers.flatten(),
        layers.linear(output_size=150),
        layers.relu(),
        layers.linear(output_size=75),
        layers.relu(),
        layers.linear(output_size=50),
        layers.relu(),
        layers.linear(output_size=nb_output_classes),
        layers.softmax(),
    ]

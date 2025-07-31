import torch
import math

from torch.nn import Module

from typing import Union

from qml.utils import make_2d_tuple
from qml.models.layers.utils import register


@register
class Conv2dBuilder:
    TYPE = "conv2d"

    @classmethod
    def elem(
        cls,
        kernel: Union[int, tuple],
        output_chan: Union[int, str] = 1,
        stride: Union[int, tuple] = (1, 1),
        dilation: Union[int, tuple] = (1, 1),
        padding: Union[int, tuple] = (0, 0),
        padding_mode: str = "zeros",
        bias: bool = True,
        groups=1,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            kernel=make_2d_tuple(kernel),
            output_channels=output_chan,
            stride=make_2d_tuple(stride),
            dilation=make_2d_tuple(dilation),
            padding=make_2d_tuple(padding),
            padding_mode=padding_mode,
            bias=bias,
            groups=groups,
        )

    @classmethod
    def make(
        cls,
        input_size,
        stride,
        dilation,
        padding_mode,
        bias,
        output_channels,
        padding,
        kernel,
        groups,
        **kwargs,
    ) -> Module:
        input_channel = input_size[0]

        module = torch.nn.Conv2d(
            in_channels=input_channel,
            out_channels=output_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            groups=groups,
        )

        return module

    @classmethod
    def predict_size(
        cls, input_size, stride, dilation, padding, kernel, output_channels, **kwargs
    ) -> tuple:
        output_h = math.floor(
            (input_size[1] + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1)
            / stride[0]
            + 1
        )
        output_w = math.floor(
            (input_size[2] + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1)
            / stride[1]
            + 1
        )

        return (output_channels, output_h, output_w)

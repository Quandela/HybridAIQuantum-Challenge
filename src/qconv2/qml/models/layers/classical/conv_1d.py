import torch
import math

from torch.nn import Module
from qml.models.layers.utils import register


@register
class Conv1dBuilder:
    TYPE = "conv1d"

    @classmethod
    def elem(
        cls,
        kernel: int,
        output_chan: int = 1,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        padding_mode: str = "zeros",
        bias: bool = True,
        groups=1,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            kernel=kernel,
            output_channels=output_chan,
            stride=stride,
            dilation=dilation,
            padding=padding,
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

        module = torch.nn.Conv1d(
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
        output_s = math.floor(
            (input_size[1] + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
        )

        return (output_channels, output_s)

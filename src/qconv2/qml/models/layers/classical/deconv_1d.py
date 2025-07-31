import torch

from torch.nn import Module
from typing import Union

from qml.models.layers.utils import register


@register
class Deconv1dBuilder:
    TYPE = "deconv1d"

    @classmethod
    def elem(
        cls,
        kernel: int,
        output_chan: int = 1,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        padding_mode: str = "zeros",
        bias: bool = True,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            kernel=kernel,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode=padding_mode,
            output_channels=output_chan,
            output_padding=output_padding,
            bias=bias,
        )

    @classmethod
    def make(
        cls,
        input_size,
        stride,
        dilation,
        padding,
        padding_mode,
        bias,
        kernel,
        output_channels,
        **kwargs,
    ) -> Module:
        input_channel = input_size[0]

        module = torch.nn.ConvTranspose1d(
            in_channels=input_channel,
            out_channels=output_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
        )

        return module

    @classmethod
    def predict_size(
        cls,
        input_size,
        stride,
        dilation,
        padding,
        kernel,
        output_channels,
        output_padding,
        **kwargs,
    ) -> tuple:
        output_s = (
            (input_size[1] - 1) * stride
            - 2 * padding
            + dilation * (kernel - 1)
            + output_padding
            + 1
        )

        return (output_channels, output_s)

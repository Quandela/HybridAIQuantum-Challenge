import torch

from torch.nn import Module
from typing import Union

from qml.models.layers.utils import register
from qml.utils import make_2d_tuple


class SubPixelConv2d(Module):
    def __init__(
        self,
        kernel_size: tuple,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        stride: tuple,
        dilation: tuple,
        padding: tuple,
        padding_mode: str,
        bias: bool,
    ):
        """
        Based on https://arxiv.org/pdf/1609.05158.pdf
        """

        super().__init__()

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * scale_factor**2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.pixel_shuffle = torch.nn.PixelShuffle(upscale_factor=scale_factor)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))


@register
class SubpixelConv2dBuilder:
    TYPE = "subpixelconv_2d"

    @classmethod
    def elem(
        cls,
        kernel: Union[int, tuple],
        output_chan: int = 3,
        scale_factor: int = 2,
        stride: tuple = (1, 1),
        dilation: tuple = (1, 1),
        padding: tuple = (0, 0),
        padding_mode: str = "zeros",
        bias: bool = True,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            output_channels=output_chan,
            kernel=make_2d_tuple(kernel),
            stride=make_2d_tuple(stride),
            dilation=make_2d_tuple(dilation),
            padding=make_2d_tuple(padding),
            padding_mode=padding_mode,
            bias=bias,
            scale_factor=scale_factor,
        )

    @classmethod
    def make(
        cls,
        input_size,
        scale_factor,
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

        return SubPixelConv2d(
            in_channels=input_channel,
            out_channels=output_channels,
            scale_factor=scale_factor,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
        )

    @classmethod
    def predict_size(cls, input_size, scale_factor, output_channels, **kwargs) -> tuple:
        return (
            output_channels,
            input_size[1] * scale_factor,
            input_size[2] * scale_factor,
        )

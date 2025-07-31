import torch
import math

from torch.nn import Module

from typing import Union

from qml.utils import make_2d_tuple
from qml.models.layers.utils import register


@register
class AvgPool2dBuilder:
    TYPE = "avgpool2d"

    @classmethod
    def elem(
        cls,
        kernel: Union[int, tuple],
        stride: Union[int, tuple] = (0, 0),
        padding: Union[int, tuple] = (0, 0),
        count_include_pad: bool = True,
        ceil_mode: bool = False,
        divisor_override: bool = None,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            kernel=make_2d_tuple(kernel),
            stride=make_2d_tuple(stride),
            padding=make_2d_tuple(padding),
            ceil_mode=ceil_mode,
            padding_mode=count_include_pad,
            divise_factor=divisor_override,
        )

    @classmethod
    def make(
        cls,
        stride,
        padding,
        kernel,
        ceil_mode,
        divise_factor,
        padding_mode,
        **kwargs,
    ) -> Module:
        divisor = divise_factor
        include_padding = padding_mode

        module = torch.nn.AvgPool2d(
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            divisor_override=divisor,
            ceil_mode=ceil_mode,
            count_include_pad=include_padding,
        )

        return module

    @classmethod
    def predict_size(
        cls, input_size, stride, padding, kernel, output_channels, **kwargs
    ) -> tuple:
        output_h = math.floor(
            (input_size[1] + 2 * padding[0] - kernel[0]) / stride[0] + 1
        )
        output_w = math.floor(
            (input_size[2] + 2 * padding[1] - kernel[1]) / stride[1] + 1
        )

        return (output_channels, output_h, output_w)

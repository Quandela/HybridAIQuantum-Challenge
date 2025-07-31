import torch

from torch.nn import Module

from typing import Union

from qml.utils import single_to_int
from qml.models.layers.utils import register


@register
class LinearBuilder:
    TYPE = "linear"

    @classmethod
    def elem(cls, output_size: Union[int, str], bias: bool = True) -> dict:
        return dict(type=cls.TYPE, output_size=single_to_int(output_size), bias=bias)

    @classmethod
    def make(cls, input_size, output_size, bias, **kwargs) -> Module:
        return torch.nn.Linear(single_to_int(input_size), output_size, bias=bias)

    @classmethod
    def predict_size(cls, output_size, **kwargs) -> tuple:
        return (output_size,)

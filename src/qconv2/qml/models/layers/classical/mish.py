import torch

from torch.nn import Module

from qml.models.layers.utils import register


@register
class MishBuilder:
    TYPE = "mish"

    @classmethod
    def elem(cls, inplace: bool = False) -> dict:
        return dict(type=cls.TYPE, inplace=inplace)

    @classmethod
    def make(cls, inplace, **kwargs) -> Module:
        return torch.nn.Mish(inplace)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size

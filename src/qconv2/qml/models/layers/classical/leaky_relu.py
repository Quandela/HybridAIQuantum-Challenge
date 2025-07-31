import torch

from torch.nn import Module

from qml.models.layers.utils import register


@register
class LeakyReLUBuilder:
    TYPE = "lrelu"

    @classmethod
    def elem(cls, alpha: float = 0.01, inplace: bool = False) -> dict:
        return dict(type=cls.TYPE, alpha=alpha, inplace=inplace)

    @classmethod
    def make(cls, alpha, inplace, **kwargs) -> Module:
        return torch.nn.LeakyReLU(alpha, inplace)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size

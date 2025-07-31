import torch

from typing import Optional

from torch.nn import Module

from qml.models.layers.utils import register


class Sum(Module):
    def __init__(self, dim: Optional[int] = None):
        super().__init__()
        assert dim >= 0, "dim has to be positive or None"

        self.dim = dim

    def forward(self, x):
        return torch.sum(x, dim=self.dim)


@register
class SumBuilder:
    TYPE = "sum"

    @classmethod
    def elem(cls, dim: Optional[int] = None) -> dict:
        return dict(type=cls.TYPE, dim=dim)

    @classmethod
    def make(cls, dim, **kwargs) -> Module:
        dim = None if dim is None else dim + 1
        return Sum(dim=dim)

    @classmethod
    def predict_size(cls, input_size, sequences, **kwargs) -> tuple:
        return input_size

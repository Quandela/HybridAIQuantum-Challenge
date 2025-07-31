import torch

from torch.nn import Module
from functools import reduce
from operator import mul

from qml.models.layers.utils import register


class Flatten(Module):
    def __init__(self, start_dim: int, end_dim: int):
        super().__init__()
        assert start_dim >= 0, "dim has to be positive or None"

        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return torch.flatten(x, start_dim=self.start_dim, end_dim=self.end_dim)


@register
class FlattenBuilder:
    TYPE = "flatten"

    @classmethod
    def elem(cls, start_dim: int = 0, end_dim: int = -1) -> dict:
        return dict(type=cls.TYPE, start_dim=start_dim, end_dim=end_dim)

    @classmethod
    def make(cls, start_dim: int, end_dim: int, **kwargs) -> Module:
        start_dim = start_dim + 1  # shift because of batch at dim 0
        end_dim = (
            end_dim if end_dim == -1 else end_dim + 1
        )  # shift because of batch at dim 0

        return Flatten(start_dim=start_dim, end_dim=end_dim)

    @classmethod
    def predict_size(cls, input_size, start_dim: int, end_dim: int, **kwargs) -> tuple:
        if end_dim != -1:
            select = input_size[start_dim:end_dim]
        else:
            select = input_size[start_dim:]

        size = reduce(mul, select)  # multiply all elements

        return (size,)

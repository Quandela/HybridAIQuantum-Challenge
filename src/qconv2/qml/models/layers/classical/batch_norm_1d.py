import torch

from torch.nn import Module
from qml.models.layers.utils import register


@register
class BatchNorm1dBuilder:
    TYPE = "batchnorm1d"

    @classmethod
    def elem(cls, eps: float = 1e-5, momentum: float = 0.1) -> dict:
        return dict(type=cls.TYPE, epsilon=eps, momentum=momentum)

    @classmethod
    def make(cls, input_size, epsilon, momentum, **kwargs) -> Module:
        return torch.nn.BatchNorm1d(input_size, eps=epsilon, momentum=momentum)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size

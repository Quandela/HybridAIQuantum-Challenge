import torch

from torch.nn import Module
from qml.models.layers.utils import register


@register
class BatchNorm2dBuilder:
    TYPE = "batchnorm2d"

    @classmethod
    def elem(cls, eps: float = 1e-5, momentum: float = 0.1) -> dict:
        return dict(type=cls.TYPE, epsilon=eps, momentum=momentum)

    @classmethod
    def make(cls, input_size, epsilon, momentum, **kwargs) -> Module:
        return torch.nn.BatchNorm2d(input_size[0], eps=epsilon, momentum=momentum)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size

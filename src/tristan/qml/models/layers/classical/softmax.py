import torch

from torch.nn import Module

from qml.models.layers.utils import register


@register
class SoftmaxBuilder:
    TYPE = "softmax"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, **kwargs) -> Module:
        return torch.nn.Softmax()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size

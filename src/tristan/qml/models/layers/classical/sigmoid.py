import torch

from torch.nn import Module

from qml.models.layers.utils import register


@register
class SigmoidBuilder:
    TYPE = "sigmoid"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, **kwargs) -> Module:
        return torch.nn.Sigmoid()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size

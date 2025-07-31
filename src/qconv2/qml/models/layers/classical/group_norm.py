import torch

from torch.nn import Module
from qml.models.layers.utils import register


@register
class GroupNormBuilder:
    TYPE = "group_norm"

    @classmethod
    def elem(cls, num_groups: int, num_channels: int) -> dict:
        return dict(type=cls.TYPE, num_groups=num_groups, num_channels=num_channels)

    @classmethod
    def make(cls, input_size, num_groups, num_channels, **kwargs) -> Module:
        return torch.nn.GroupNorm(num_groups, num_channels)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size

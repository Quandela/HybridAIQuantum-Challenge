import torch

from torch.nn import Module

from qml.models.layers.utils import register


@register
class PixelShuffleBuilder:
    TYPE = "pixel_shuffle"

    @classmethod
    def elem(cls, factor: int) -> dict:
        return dict(type=cls.TYPE, factor=factor)

    @classmethod
    def make(cls, factor, **kwargs) -> Module:
        return torch.nn.PixelShuffle(factor)

    @classmethod
    def predict_size(cls, input_size, factor, **kwargs) -> tuple:
        c_in, h_in, w_in = input_size[-3:]

        ouput_size = tuple(
            [
                *input_size[:-3],
                int(c_in / factor / factor),
                h_in * factor,
                w_in * factor,
            ]
        )

        return ouput_size

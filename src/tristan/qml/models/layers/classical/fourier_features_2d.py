import torch

from torch.nn import Module

from qml.models.layers.utils import register


def fourier_features_2d(x: torch.Tensor, weight: torch.Tensor):
    # Gaussian Fourier feature mapping.
    # https://arxiv.org/abs/2006.10739
    # https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    b, c, h, w = x.shape

    x = x.permute(0, 2, 3, 1).reshape(b * w * h, c)
    x = x @ weight
    x = x.view(b, h, w, weight.shape[1])
    x = x.permute(0, 3, 1, 2)
    x = 2 * torch.pi * x
    x = torch.cat([torch.sin(x), torch.cos(x)], dim=1)

    return x


class FourierFeatures2d(Module):
    def __init__(self, input_chan: int, output_chan: int, sigma: float):
        super().__init__()

        assert output_chan % 2 == 0

        self._input_chan = input_chan
        self._mapping_size = output_chan // 2
        self.weight = torch.nn.Parameter(
            torch.randn((self._input_chan, self._mapping_size)) * sigma
        )

    def forward(self, x):
        return fourier_features_2d(x, self.weight.detach())


@register
class FourierFeatures2dBuilder:
    TYPE = "fourier_features_2d"

    @classmethod
    def elem(cls, output_chan: int, sigma: float = 0.1) -> dict:
        return dict(type=cls.TYPE, output_chan=output_chan, sigma=sigma)

    @classmethod
    def make(cls, input_size, output_chan: int, sigma: float, **kwargs) -> Module:
        return FourierFeatures2d(input_size[0], output_chan, sigma)

    @classmethod
    def predict_size(cls, input_size, output_chan, **kwargs) -> tuple:
        return (output_chan, input_size[1], input_size[2])

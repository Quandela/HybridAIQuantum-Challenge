import torch

from torch.nn import Module
from qml.models.layers.utils import register


@register
class PReLUBuilder:
    TYPE = "prelu"

    @classmethod
    def elem(cls, nb_params: int) -> dict:
        return dict(type=cls.TYPE, nb_params=nb_params)

    @classmethod
    def single(cls) -> dict:
        return cls.elem(1)

    @classmethod
    def multi(cls) -> dict:
        return cls.elem(-1)

    @classmethod
    def make(cls, input_size, nb_params=1, **kwargs) -> Module:
        if nb_params != 1:
            nb_params = input_size

            if isinstance(nb_params, tuple) and len(nb_params) == 3:
                nb_params = nb_params[0]

        return torch.nn.PReLU(num_parameters=nb_params)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size

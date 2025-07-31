import abc
import torch
import perceval as pcvl

from typing import Tuple
from collections import Counter


from qml.models.layers.quantum.converters.utils import (
    thresholded_size,
    pnr_size,
    generate_states,
    generate_states_multi_photons,
)
from qml.models.layers.quantum.converters.register import register, converter_factory


def build_converter(name: str, **kwargs):
    return converter_factory()[name].make(**kwargs)


def predict_size(name: str, input_size: torch.Size, **kwargs) -> Tuple:
    return converter_factory()[name].predict_size(input_size, **kwargs)


class BSConverter(abc.ABC):
    @abc.abstractmethod
    def to_tensor(distribution: pcvl.BSDistribution, start_n: int, min_n: int):
        pass


class TopKConverter(BSConverter):
    def __init__(self, k: int):
        super().__init__()

        assert k > 0

        self._k = k

    def to_tensor(
        self, distribution: pcvl.BSDistribution, start_n: int, min_n: int
    ) -> torch.Tensor:
        # Finding k highest values
        dist_counter = Counter(distribution)
        highest = dist_counter.most_common(self._k)

        t = torch.zeros(size=(self._k, distribution.m))

        i = 0
        for k, v in highest:
            for j in range(distribution.m):
                t[i][j] = k[j]
            i += 1

        return t.reshape(1, -1)


@register
class TopKConverterBuilder:
    TYPE = "topk"

    @classmethod
    def make(cls, k, **kwargs) -> TopKConverter:
        return TopKConverter(k=k)

    @classmethod
    def predict_size(cls, input_size, k, **kwargs) -> Tuple:
        return (
            input_size[0],
            k,
        )


@register
class BestConverterBuilder:
    TYPE = "best"

    @classmethod
    def make(cls, **kwargs) -> TopKConverter:
        return TopKConverter(k=1)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> Tuple:
        return (input_size[0],)


class CumulativeModeConverter(BSConverter):
    def __init__(self, thresholded_output):
        super().__init__()
        self._thresholded_output = thresholded_output

    def to_tensor(
        self, distribution: pcvl.BSDistribution, start_n: int, min_n: int
    ) -> torch.Tensor:
        t = torch.zeros(size=(distribution.m, 1))

        for state, prob in distribution.items():
            for i, photons in enumerate(state):
                t[i] += photons * prob

        t = torch.div(t, start_n)

        return t.reshape(1, -1)


@register
class CumulativeModeBuilder:
    TYPE = "cumulative_mode"

    @classmethod
    def make(cls, thresholded_output, **kwargs) -> CumulativeModeConverter:
        return CumulativeModeConverter(thresholded_output)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> Tuple:
        return (input_size[0],)


class DefaultConverter(BSConverter):
    def __init__(self, thresholded_output):
        super().__init__()
        self._thresholded_output = thresholded_output

    def to_tensor(
        self, distribution: pcvl.BSDistribution, start_n: int, min_n: int
    ) -> torch.Tensor:
        """Transforms the perceval results into a list of probabilities, where each output is always represented at the same position"""

        # First, we generate a list of all possible output states
        if self._thresholded_output:
            state_list = generate_states(distribution.m, start_n, min_n)
            size = thresholded_size(distribution.m, start_n, min_n)
        else:
            state_list = generate_states_multi_photons(distribution.m, start_n)
            size = pnr_size(distribution.m, start_n, min_n)

        # Then we take the probabilities from the BSD in the order of the list
        t = torch.zeros(size)
        for i, state in enumerate(state_list):
            t[i] = distribution[state]

        return t.reshape(1, -1)


@register
class DefaultConverterBuilder:
    TYPE = "default"

    @classmethod
    def make(cls, thresholded_output, **kwargs) -> DefaultConverter:
        return DefaultConverter(thresholded_output)

    @classmethod
    def predict_size(cls, input_size, thresholded_output, **kwargs) -> Tuple:
        if thresholded_output:
            return (thresholded_size(*input_size),)
        k = pnr_size(*input_size)
        return (k,)

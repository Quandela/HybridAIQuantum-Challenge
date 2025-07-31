from torch.nn import Module

from qml.models.layers.utils import build_topology_from_list, predict_size, register


class Repeat(Module):
    def __init__(self, sequence, iterations):
        super().__init__()
        assert sequence
        assert int(iterations)

        self.sequence = sequence
        self.iterations = iterations

    def forward(self, x):
        for _ in range(0, self.iterations):
            x = self.sequence(x)

        return x


@register
class RepeatBuilder:
    TYPE = "repeat"

    @classmethod
    def elem(cls, sequence: list, iterations: int) -> dict:
        assert int(iterations)
        assert sequence
        return dict(type=cls.TYPE, sequence=sequence, iterations=iterations)

    @classmethod
    def make(cls, input_size, sequence, iterations, **kwargs) -> Module:
        seq, output_size = build_topology_from_list(
            sequence=sequence,
            input_size=input_size,
        )

        return Repeat(seq, iterations)

    @classmethod
    def predict_size(cls, input_size, sequence, **kwargs) -> tuple:
        return predict_size(sequence=sequence, input_size=input_size)

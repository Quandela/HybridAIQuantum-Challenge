from typing import List

from torch.nn import Module

from qml.models.layers.utils import build_topology_from_list, predict_size, register


class Mul(Module):
    def __init__(self, sequences):
        super().__init__()
        assert sequences, "sequences has to be a non empty list"

        self.sequences = sequences

        for idx, sequence in enumerate(sequences):
            self.add_module(name=str(idx), module=sequence)

    def forward(self, x):
        res = self.sequences[0].forward(x)
        for sequence in self.sequences[1:]:
            res = res * sequence.forward(x)
        return res


@register
class MulBuilder:
    TYPE = "mul"

    @classmethod
    def elem(cls, sequences: List[list]) -> dict:
        return dict(type=cls.TYPE, sequences=sequences)

    @classmethod
    def make(cls, input_size, sequences, context, **kwargs) -> Module:
        assert sequences, "sequences has to be a non empty list"

        built_sequences = []
        for sequence in sequences:
            built_sequence, output_size = build_topology_from_list(
                sequence=sequence,
                input_size=input_size,
                context=context,
            )
            built_sequences.append(built_sequence)

        # output shape is supposed to be the same for each sub sequence
        return Mul(built_sequences)

    @classmethod
    def predict_size(cls, input_size, sequences, **kwargs) -> tuple:
        assert sequences, "sequences has to be a non empty list"
        return predict_size(sequence=sequences[0], input_size=input_size)

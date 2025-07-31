import torch

from torch.nn import Module

from qml.models.layers.utils import build_topology_from_list, predict_size, register


class Concat(Module):
    def __init__(self, sequences, dim: int):
        super().__init__()
        assert sequences, "sequences has to be a non empty list"
        assert dim >= 0, "dim has to be positive or None"

        self.sequences = sequences
        self.dim = dim

        for idx, sequence in enumerate(sequences):
            self.add_module(name=str(idx), module=sequence)

    def forward(self, x):
        # Calculate results
        seq_result = []
        for i in range(0, len(self.sequences)):
            seq_result.append(self.sequences[i](x))

        # Concat and return
        return torch.cat(seq_result, dim=self.dim)


@register
class ConcatBuilder:
    TYPE = "concat"

    @classmethod
    def elem(cls, sequences: list, dim: int = 0) -> dict:
        if not isinstance(sequences, list):
            sequences = [sequences]

        for i in range(len(sequences)):
            sub_sequence = sequences[i]
            if not isinstance(sub_sequence, list):
                sequences[i] = [sub_sequence]

        return dict(type=cls.TYPE, sequences=sequences, dim=dim)

    @classmethod
    def make(cls, input_size, sequences, dim, **kwargs) -> Module:
        assert sequences, "sequences has to be a non empty list"
        assert dim >= 0, "dim has to be positive or None"

        # 0 is for batch
        actual_dim = dim + 1

        built_sequences = []

        for sequence in sequences:
            built_sequence, output_size = build_topology_from_list(
                sequence=sequence,
                input_size=input_size,
            )
            built_sequences.append(built_sequence)

        # output shape is supposed to be the same for each sub sequence
        return Concat(built_sequences, actual_dim)

    @classmethod
    def predict_size(cls, input_size, sequences, dim, **kwargs) -> tuple:
        assert sequences, "sequences has to be a non empty list"
        assert dim >= 0, "dim has to be positive or None"

        dim_cat = 0
        for sequence in sequences:
            output_size = predict_size(sequence=sequence, input_size=input_size)
            dim_cat += output_size[dim]

        output_size = list(output_size)
        output_size[dim] = dim_cat
        output_size = tuple(output_size)

        return output_size

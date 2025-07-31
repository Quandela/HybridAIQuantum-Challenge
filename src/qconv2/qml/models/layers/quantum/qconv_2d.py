import torch
import math

from torch.nn import Module

from typing import Union, Tuple

from qml.utils import make_2d_tuple
from qml.models.layers.utils import register

from qml.models.layers.quantum.ansatz import build_ansatz
from qml.models.layers.quantum.feature_maps import build_feature_map
from qml.models.layers.quantum.slos import QuantumLayer, OutputMappingStrategy

# from joblib import Parallel, delayed


class QuantumConv2d(Module):
    def __init__(
        self,
        fmap_name: str,
        ansatz_name: str,
        input_size: Tuple,
        kernel_size: Tuple,
        stride: Tuple,
        dilation: Tuple,
        padding: Tuple,
        # False by default. True to map inputs to multiple features.
        # TODO: remove this and have a cleaner generic implementation
        advanced_input_mapping: bool,
    ):
        super().__init__()

        nb_modes = kernel_size[0] * kernel_size[1]

        self._output_channels = nb_modes
        init_input = torch.ones(nb_modes)

        fmap_params = build_feature_map(fmap_name, init_input)
        ansatz_circuit = build_ansatz(ansatz_name, init_input, None)

        circuit = fmap_params.circuit
        circuit.add(0, ansatz_circuit)

        trainable_parameters = list(ansatz_circuit.params)
        input_state = list(fmap_params.input_state)

        self._output_size = QuantumConv2dBuilder.predict_size(
            input_size, stride, dilation, padding, kernel_size
        )

        self._unfold_module = torch.nn.Unfold(kernel_size, dilation, padding, stride)

        self._advanced_input_mapping = advanced_input_mapping
        input_size = nb_modes * 3 if advanced_input_mapping else nb_modes

        self._slos_module = QuantumLayer(
            input_size=input_size,
            output_size=None,  # Deduced
            no_bunching=False,
            circuit=circuit,
            trainable_parameters=trainable_parameters,
            output_mapping_strategy=OutputMappingStrategy.NONE,
            input_state=input_state,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # N, H, W
        if self._advanced_input_mapping:
            x_2pi = 2 * torch.pi * x
            x_cat = torch.cat((-x_2pi, x_2pi, torch.sin(x_2pi)), dim=1)
        else:
            x_cat = (x - 0.5) * torch.pi / 2

        # N, K*K, Ho*Wo
        x_unfolded = self._unfold_module(x_cat)

        # Ho*Wo
        nb_blocks = x_unfolded.shape[-1]

        # N, K*K, Ho*Wo
        y_unfolded = torch.Tensor(x.shape[0], self._output_channels, nb_blocks).to(
            x.device
        )

        for x_block_idx in range(nb_blocks):
            # N, K*K
            x_block = x_unfolded[:, :, x_block_idx]

            y_block_distrib_keys, y_block_distrib_probs = self._slos_module.forward(
                x_block
            )

            ### Output - ARGMAX
            # Get the maximal distribution from its probability (from 0 to 1)
            y_max_distrib_idx = torch.argmax(y_block_distrib_probs, dim=1)

            # Get the maximal distribution key (1,0,1,0,0...)
            y_max_distrib_key = [y_block_distrib_keys[i] for i in y_max_distrib_idx]
            y_block = torch.Tensor(list(y_max_distrib_key))

            ### Output - CUMULATIVE
            # # Sum probabilities
            # y_tensors_keys = torch.Tensor(list(y_block_distrib_keys))
            # y_norm_factor = torch.Tensor(list(y_block_distrib_keys[0])).sum()

            # y_block = torch.mm(
            #     y_block_distrib_probs, y_tensors_keys
            # ) / y_norm_factor

            # print("distrib", len(y_block_distrib_keys), y_block_distrib_keys)
            # print("keys", y_tensors_keys.shape, y_tensors_keys)
            # print("probs", y_block_distrib_probs.shape, y_block_distrib_probs)
            # print("block", y_block.shape, y_block)

            # N, K*K, Ho*Wo
            y_unfolded[:, :, x_block_idx] = y_block

        # N, K*K, Ho, Wo
        return y_unfolded.view(-1, *self._output_size)


@register
class QuantumConv2dBuilder:
    TYPE = "qconv2d"

    @classmethod
    def elem(
        cls,
        fmap: str,
        ansatz: str,
        kernel: Union[int, tuple],
        stride: Union[int, tuple] = (1, 1),
        dilation: Union[int, tuple] = (1, 1),
        padding: Union[int, tuple] = (0, 0),
        advanced_input_mapping: bool = False,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            fmap_name=fmap,
            ansatz_name=ansatz,
            kernel=make_2d_tuple(kernel),
            stride=make_2d_tuple(stride),
            dilation=make_2d_tuple(dilation),
            padding=make_2d_tuple(padding),
            advanced_input_mapping=advanced_input_mapping,
        )

    @classmethod
    def make(
        cls,
        fmap_name,
        ansatz_name,
        input_size,
        stride,
        dilation,
        padding,
        kernel,
        advanced_input_mapping,
        **kwargs,
    ) -> Module:
        module = QuantumConv2d(
            input_size=input_size,
            fmap_name=fmap_name,
            ansatz_name=ansatz_name,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            advanced_input_mapping=advanced_input_mapping,
        )

        return module

    @classmethod
    def predict_size(
        cls,
        input_size,
        stride,
        dilation,
        padding,
        kernel,
        **kwargs,
    ) -> tuple:
        output_mode = kernel[0] * kernel[1]

        output_h = math.floor(
            (input_size[1] + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1)
            / stride[0]
            + 1
        )
        output_w = math.floor(
            (input_size[2] + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1)
            / stride[1]
            + 1
        )

        return (output_mode, output_h, output_w)

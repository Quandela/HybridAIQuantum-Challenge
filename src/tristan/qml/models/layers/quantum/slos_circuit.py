import torch

from torch.nn import Module, Parameter

from typing import Optional, Tuple

from qml.models.layers.utils import register
from qml.models.layers.quantum.ansatz import build_ansatz
from qml.models.layers.quantum.feature_maps import build_feature_map

from qml.models.layers.quantum.slos import QuantumLayer, OutputMappingStrategy


class SLOSCircuit(Module):
    def __init__(
        self,
        fmap_name: str,
        ansatz_name: Optional[str] = None,
        input_size: Optional[Tuple] = None,
        output_size: Optional[Tuple] = None,
    ):
        super().__init__()

        self._fmap_name = fmap_name
        self._ansatz_name = ansatz_name
        self._ansatz_circuit = None

        init_input = torch.ones(input_size)

        fmap_params = build_feature_map(self._fmap_name, init_input)
        circuit = fmap_params.circuit

        trainable_parameters = None
        input_state = list(fmap_params.input_state)

        if self._ansatz_name:
            self._ansatz_circuit = build_ansatz(self._ansatz_name, init_input, None)
            circuit.add(0, self._ansatz_circuit)

            trainable_parameters = list(self._ansatz_circuit.params)

        self._slos_module = QuantumLayer(
            input_size=input_size[0],
            output_size=output_size,
            no_bunching=False,
            circuit=circuit,
            trainable_parameters=trainable_parameters,
            output_mapping_strategy=OutputMappingStrategy.GROUPING,
            input_state=input_state,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, d = self._slos_module.forward(x)
        return d


@register
class SLOSCircuitBuilder:
    TYPE = "slos_circuit"

    @classmethod
    def elem(
        cls,
        fmap: str,
        output_size: int,
        ansatz: Optional[str] = None,
        # converter: Optional[str] = "",
    ) -> dict:
        return dict(
            type=cls.TYPE,
            ansatz_name=ansatz,
            fmap_name=fmap,
            output_size=output_size,
        )

    @classmethod
    def make(
        cls,
        fmap_name,
        ansatz_name,
        input_size,
        output_size,
        **kwargs,
    ) -> Module:
        return SLOSCircuit(
            fmap_name=fmap_name,
            ansatz_name=ansatz_name,
            input_size=input_size,
            output_size=output_size,
        )

    @classmethod
    def predict_size(
        cls,
        output_size,
        **kwargs,
    ) -> tuple:
        return (output_size,)

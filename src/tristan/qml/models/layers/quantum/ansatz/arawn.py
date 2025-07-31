import torch
import perceval as pcvl

from typing import Tuple

from qml.models.layers.quantum.ansatz.register import register


@register
class ArawnBuilder:
    TYPE = "arawn"

    @classmethod
    def make(cls, x: torch.Tensor, weights: torch.Tensor) -> pcvl.Circuit:
        x = x.squeeze(0)  # batch size = 1
        modes_count = x.shape[0]
        circuit = pcvl.Circuit(modes_count)

        j = 0
        for i in range(modes_count - 1):
            t_val = float(weights[j])
            circuit.add(i, pcvl.BS.Rx(theta=t_val))
            circuit.add(i, pcvl.PS(phi=t_val))
            j += 1

        for i in range(modes_count - 1):
            t_val = float(weights[j])
            circuit.add(i, pcvl.BS.H(theta=t_val))
            circuit.add(i, pcvl.PS(phi=t_val))
            j += 1

        for i in range(modes_count - 1):
            t_val = float(weights[j])
            circuit.add(i, pcvl.BS.Ry(theta=t_val))
            circuit.add(i, pcvl.PS(phi=t_val))
            j += 1

        return circuit

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> Tuple:
        # Ansatz always follows a feature_map, just return the same size
        # (Mode, nb photon, min_photon)
        return input_size

    @classmethod
    def make_weights(cls, input_size, iterations, **kwargs) -> torch.Tensor:
        nb_parameters = input_size[0] * 3
        return torch.FloatTensor(size=(iterations, nb_parameters)).uniform_(
            -torch.pi, torch.pi
        )

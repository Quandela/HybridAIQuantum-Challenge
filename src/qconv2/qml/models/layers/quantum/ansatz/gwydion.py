import torch
import perceval as pcvl

from typing import Tuple

from qml.models.layers.quantum.ansatz.register import register


@register
class GwydionBuilder:
    TYPE = "gwydion"
    FRONT_BLOCK_SIZE = 1
    BACK_BLOCK_SIZE = 2

    @classmethod
    def make(cls, x: torch.Tensor, weights: torch.Tensor) -> pcvl.Circuit:
        x = x.squeeze(0)  # batch size = 1
        modes_count = x.shape[0]
        circuit = pcvl.Circuit(modes_count)

        weight_idx = 0

        def _w() -> float:
            nonlocal weight_idx
            val = float(weights[weight_idx])
            weight_idx += 1
            return val

        # Front variational components
        for i in range(modes_count // 2 - 1):
            circuit.add(i, pcvl.BS(theta=_w()))

        for i in reversed(range(modes_count // 2, modes_count - 1)):
            circuit.add(i, pcvl.BS(theta=_w()))

        circuit.add(modes_count // 2 - 1, pcvl.BS(theta=_w()))

        for i in reversed(range(modes_count // 2 - 1)):
            circuit.add(i, pcvl.BS(theta=_w()))

        for i in range(modes_count // 2, modes_count - 1):
            circuit.add(i, pcvl.BS(theta=_w()))

        # Feature map
        t = x * torch.pi
        for i in range(modes_count):
            p_val = float(t[i])
            circuit.add(i, pcvl.PS(p_val))

        # Back variational components
        for _ in range(modes_count // 2):
            for i in range(0, modes_count - 1, 2):
                circuit.add(i, pcvl.BS(theta=_w()).add(0, pcvl.PS(phi=_w())))

            for i in range(1, modes_count - 1, 2):
                circuit.add(i, pcvl.BS(theta=_w()).add(0, pcvl.PS(phi=_w())))

        return circuit

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> Tuple:
        # Ansatz always follows a feature_map, just return the same size
        # (Mode, nb photon, min_photon)
        return input_size

    @classmethod
    def make_weights(cls, input_size, iterations, **kwargs) -> torch.Tensor:
        nb_modes = input_size[0]

        nb_front_parameters = GwydionBuilder.FRONT_BLOCK_SIZE * (2 * nb_modes - 1)

        depth_block_size = nb_modes
        nb_back_parameters = (
            GwydionBuilder.BACK_BLOCK_SIZE
            * depth_block_size
            // 2
            * (depth_block_size - 1)
        )

        nb_parameters = nb_front_parameters + nb_back_parameters

        return torch.FloatTensor(size=(iterations, nb_parameters)).uniform_(
            -torch.pi, torch.pi
        )

import torch
import perceval as pcvl

from typing import Tuple

from qml.models.layers.quantum.ansatz.register import register


@register
class GofanonBuilder:
    TYPE = "gofanon"
    BACK_BLOCK_SIZE = 2

    @classmethod
    def make(cls, x: torch.Tensor, weights: torch.Tensor) -> pcvl.Circuit:
        x = x.squeeze(0)  # batch size = 1
        modes_count = x.shape[0]
        circuit = pcvl.Circuit(modes_count)

        weight_idx = 0

        def _w() -> float:
            nonlocal weight_idx
            if weights:
                val = float(weights[weight_idx])
            else:
                val = pcvl.P(f"phi={weight_idx}")
            weight_idx += 1
            return val

        for _ in range(modes_count // 2):
            for i in range(0, modes_count - 1, 2):
                circuit.add(
                    i,
                    pcvl.BS(theta=_w())
                    .add(0, pcvl.PS(phi=_w()))
                    .add(1, pcvl.PS(phi=_w())),
                )

            for i in range(1, modes_count - 1, 2):
                circuit.add(
                    i,
                    pcvl.BS.Ry(theta=_w())
                    .add(0, pcvl.PS(phi=_w()))
                    .add(1, pcvl.PS(phi=_w())),
                )

        return circuit

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> Tuple:
        # Ansatz always follows a feature_map, just return the same size
        # (Mode, nb photon, min_photon)
        return input_size

    @classmethod
    def make_weights(cls, input_size, iterations, **kwargs) -> torch.Tensor:
        nb_modes = input_size[0]

        depth_block_size = nb_modes
        nb_parameters = (
            GofanonBuilder.BACK_BLOCK_SIZE
            * depth_block_size
            // 2
            * (depth_block_size - 1)
        )

        return torch.FloatTensor(size=(iterations, nb_parameters)).uniform_(
            -torch.pi / 2, torch.pi / 2
        )

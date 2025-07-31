import torch
import perceval as pcvl

from typing import Tuple

from qml.models.layers.quantum.ansatz.register import register
from sympy import pi


@register
class PenarddunBuilder:
    TYPE = "penarddun"
    INTERFEROMETER_DEPTH = 6

    @classmethod
    def make(cls, x: torch.Tensor, weights: torch.Tensor) -> pcvl.Circuit:
        x = x.squeeze(0)  # batch size = 1
        modes_count = x.shape[0]

        def _w(idx) -> float:
            if weights:
                val = float(weights[idx])
            else:
                val = pcvl.P(f"phi={idx}")
            return val

        circuit = pcvl.GenericInterferometer(
            modes_count,
            lambda i: (
                pcvl.BS()
                .add(0, pcvl.PS(phi=_w(2 * i)))
                .add(0, pcvl.BS())
                .add(0, pcvl.PS(phi=_w(2 * i + 1)))
            ),
            shape=pcvl.InterferometerShape.RECTANGLE,
            depth=PenarddunBuilder.INTERFEROMETER_DEPTH,
        )

        # pcvl.pdisplay(circuit, output_format=None, recursive=True)

        return circuit

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> Tuple:
        # Ansatz always follows a feature_map, just return the same size
        # (Mode, nb photon, min_photon)
        return input_size

    @classmethod
    def make_weights(cls, input_size, iterations, **kwargs) -> torch.Tensor:
        nb_modes = input_size[0]

        # nb_parameters = nb_modes * (nb_modes - 1)
        nb_parameters = PenarddunBuilder.INTERFEROMETER_DEPTH * (nb_modes - 1)

        return torch.FloatTensor(size=(iterations, nb_parameters)).uniform_(
            -torch.pi / 2, torch.pi / 2
        )

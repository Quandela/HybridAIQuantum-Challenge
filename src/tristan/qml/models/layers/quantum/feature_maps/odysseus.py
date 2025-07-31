import torch
import perceval as pcvl

from typing import Tuple

from qml.models.layers.quantum.feature_maps.register import register
from qml.models.layers.quantum.feature_maps.feature_map_params import FeatureMapParams


@register
class OdysseusBuilder:
    TYPE = "odysseus"

    @classmethod
    def make(cls, x: torch.Tensor, **kwargs) -> FeatureMapParams:
        x = x.squeeze(0)
        modes_count, photons_count, min_photon = OdysseusBuilder.predict_size(x.shape)

        circuit = pcvl.Circuit(m=modes_count)

        x_idx = 0

        def _x() -> pcvl.P:
            nonlocal x_idx
            val = pcvl.P(f"x={x_idx}")
            x_idx += 1
            return val

        for i in range(modes_count // 2 - 1):
            circuit.add(i, pcvl.BS(theta=_x()))

        for i in reversed(range(modes_count // 2, modes_count - 1)):
            circuit.add(i, pcvl.BS(theta=_x()))

        for i in range(modes_count):
            circuit.add(i, pcvl.PS(phi=_x()))

        for i in range(modes_count // 2 - 1):
            circuit.add(i, pcvl.BS.Ry(theta=_x()))

        for i in reversed(range(modes_count // 2, modes_count - 1)):
            circuit.add(i, pcvl.BS.Ry(theta=_x()))

        circuit.add(modes_count - 2, pcvl.BS.Ry(theta=_x()))
        circuit.add(modes_count // 2 - 1, pcvl.BS.Ry(theta=_x()))
        circuit.add(modes_count // 2 + 1, pcvl.BS.Ry(theta=_x()))
        circuit.add(0, pcvl.BS.Ry(theta=_x()))

        input_state = modes_count * [0]
        places = torch.linspace(0, modes_count - 1, photons_count)

        for photon in places:
            input_state[int(photon)] = 1

        input_state = pcvl.BasicState(input_state)

        return FeatureMapParams(
            circuit=circuit,
            input_state=input_state,
            min_detect_photon=min_photon,
        )

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> Tuple:
        modes_count = input_size[0]
        photons_count = 4
        min_photon = photons_count

        # (modes, nb photon, min_photon)
        return (modes_count, photons_count, min_photon)

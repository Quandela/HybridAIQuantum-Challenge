import torch
import perceval as pcvl
import math

from typing import Tuple

from qml.models.layers.quantum.feature_maps.register import register
from qml.models.layers.quantum.feature_maps.feature_map_params import FeatureMapParams

photons_count = 3
modes_count = 12

# unitary = pcvl.Matrix.random_unitary(modes_count)

# mzi = (
#     pcvl.BS()
#     // (0, pcvl.PS(phi=pcvl.Parameter("φ_a")))
#     // pcvl.BS()
#     // (1, pcvl.PS(phi=pcvl.Parameter("φ_b")))
# )

# circuit = pcvl.Circuit.decomposition(
#     unitary,
#     mzi,
#     phase_shifter_fn=pcvl.PS,
#     shape=pcvl.InterferometerShape.TRIANGLE
# )

circuit = pcvl.GenericInterferometer(
    modes_count,
    lambda i: pcvl.BS()
    // pcvl.PS(i * 2 * math.pi * 2 / (modes_count * (modes_count - 1))),
    shape=pcvl.InterferometerShape.RECTANGLE,
)


@register
class TeucrosBuilder:
    TYPE = "teucros"

    @classmethod
    def make(cls, x: torch.Tensor, **kwargs) -> FeatureMapParams:
        x = x.squeeze(0)
        modes_count, photons_count, min_photon = TeucrosBuilder.predict_size(x.shape)

        input_state = modes_count * [0]
        _, places = torch.topk(x, photons_count)
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
        nb_data_to_encode = input_size[0]

        min_photon = photons_count

        return (modes_count, photons_count, min_photon)

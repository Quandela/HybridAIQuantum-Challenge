import torch
import perceval as pcvl

from typing import Tuple

from qml.models.layers.quantum.feature_maps.register import register
from qml.models.layers.quantum.feature_maps.feature_map_params import FeatureMapParams


@register
class HeleneBuilder:
    TYPE = "helene"

    @classmethod
    def make(cls, x: torch.Tensor, **kwargs) -> FeatureMapParams:
        x = x.squeeze(0)
        modes_count, photons_count, min_photon = HeleneBuilder.predict_size(x.shape)

        circuit = pcvl.Circuit(m=modes_count)

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
        nb_data_to_encode = input_size[0]

        photons_count = 4
        modes_count = nb_data_to_encode
        min_photon = photons_count

        # (modes, nb photon, min_photon)
        return (modes_count, photons_count, min_photon)

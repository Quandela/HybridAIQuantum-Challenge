import torch
import perceval as pcvl

from typing import Tuple

from qml.models.layers.quantum.feature_maps.register import register
from qml.models.layers.quantum.feature_maps.feature_map_params import FeatureMapParams


@register
class AjaxBuilder:
    TYPE = "ajax"

    @classmethod
    def make(cls, x: torch.Tensor, **kwargs) -> FeatureMapParams:
        x = x.squeeze(0)
        modes_count, photons_count, min_photon = AjaxBuilder.predict_size(x.shape)

        t = x * torch.pi * 2
        circuit = pcvl.Circuit(m=modes_count)

        data_index = 0
        for j in range(modes_count):
            for i in range(modes_count - j - 1):
                angle = float(t[data_index])
                circuit.add(i, pcvl.BS(angle).add(0, pcvl.PS(angle)))
                data_index += 1

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
        max_modes_count = 30

        # Counting the number of mode required to store all the data in a triangle circuit
        for i in range(max_modes_count):
            nb_of_components = sum(i + 1 for i in range(i))
            if nb_of_components > nb_data_to_encode:
                modes_count = i
                break

        photons_count = 3
        min_photon = photons_count

        # (modes, nb photon, min_photon)
        return (modes_count, photons_count, min_photon)

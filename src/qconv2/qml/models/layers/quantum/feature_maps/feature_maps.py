import torch

from typing import Tuple

from qml.models.layers.quantum.feature_maps.register import feature_maps_factory
from qml.models.layers.quantum.feature_maps.feature_map_params import FeatureMapParams


def build_feature_map(name: str, x: torch.Tensor) -> FeatureMapParams:
    return feature_maps_factory()[name].make(x)


def predict_size(name: str, input_size: torch.Size) -> Tuple:
    return feature_maps_factory()[name].predict_size(input_size)

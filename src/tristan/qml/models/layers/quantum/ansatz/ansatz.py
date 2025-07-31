import torch
import perceval as pcvl

from typing import Tuple

from qml.models.layers.quantum.ansatz.register import ansatz_factory


def build_ansatz(name: str, x: torch.Tensor, weights: torch.Tensor) -> pcvl.Circuit:
    return ansatz_factory()[name].make(x, weights)


def predict_size(name: str, input_size: torch.Size) -> Tuple:
    return ansatz_factory()[name].predict_size(input_size)


def build_weights(name: str, input_size: torch.Size, **kwargs) -> torch.Tensor:
    return ansatz_factory()[name].make_weights(input_size, **kwargs)

from torch.nn import Sequential

from typing import Tuple, List

from qml.utils import Factory, make_tuple


__FACTORY = Factory("layers")


def layers_factory() -> Factory:
    global __FACTORY
    return __FACTORY


def register(cls):
    return layers_factory().register(cls.TYPE)(cls)


def predict_size(sequence: List, input_size: Tuple) -> Tuple[int]:
    if not sequence:
        return input_size

    if isinstance(sequence, dict):
        sequence = sequence["sequence"]

    previous_size = input_size

    for desc_layer in sequence:
        layer_class = layers_factory()[desc_layer["type"]]
        previous_size = layer_class.predict_size(input_size=previous_size, **desc_layer)

    return previous_size


def build_topology_from_list(
    sequence: List, input_size: Tuple = (1,)
) -> Tuple[Sequential, Tuple]:
    if not sequence:
        return None

    if isinstance(sequence, dict):
        sequence = sequence["sequence"]

    modules = []
    previous_size = input_size

    for desc_layer in sequence:
        layer_class = layers_factory()[desc_layer["type"]]
        output_size = layer_class.predict_size(input_size=previous_size, **desc_layer)
        module = layer_class.make(input_size=previous_size, **desc_layer)

        previous_size = make_tuple(output_size)

        modules.append(module)

    return Sequential(*modules), previous_size

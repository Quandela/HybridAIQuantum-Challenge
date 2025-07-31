from qml.models.layers.quantum.gradients.register import factory


def build_gradient_method(name: str):
    return factory()[name]

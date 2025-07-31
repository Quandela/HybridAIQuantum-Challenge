from qml.utils import Factory

__FACTORY = Factory("topology")


def factory() -> Factory:
    global __FACTORY
    return __FACTORY


def register(name: str):
    return factory().register(name)

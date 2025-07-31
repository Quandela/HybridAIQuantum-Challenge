from qml.utils import Factory


__FACTORY = Factory("ansatz")


def ansatz_factory() -> Factory:
    global __FACTORY
    return __FACTORY


def register(cls):
    return ansatz_factory().register(cls.TYPE)(cls)

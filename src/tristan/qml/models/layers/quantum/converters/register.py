from qml.utils import Factory


__FACTORY = Factory("converters")


def converter_factory() -> Factory:
    global __FACTORY
    return __FACTORY


def register(cls):
    return converter_factory().register(cls.TYPE)(cls)

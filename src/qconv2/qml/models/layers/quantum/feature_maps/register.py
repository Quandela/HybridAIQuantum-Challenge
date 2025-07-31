from qml.utils import Factory


__FACTORY = Factory("feature_maps")


def feature_maps_factory() -> Factory:
    global __FACTORY
    return __FACTORY


def register(cls):
    return feature_maps_factory().register(cls.TYPE)(cls)

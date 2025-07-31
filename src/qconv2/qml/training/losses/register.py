from qml.utils.factory import Factory


__FACTORY = Factory("loss")


def factory() -> Factory:
    global __FACTORY
    return __FACTORY


def register(name: str):
    return factory().register(name)

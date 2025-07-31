from qml.utils import Factory


__FACTORY = Factory("runner")


def runner_factory() -> Factory:
    global __FACTORY
    return __FACTORY


def register(cls):
    return runner_factory().register(cls.TYPE)(cls)

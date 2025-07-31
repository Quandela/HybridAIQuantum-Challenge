from typing import List, Iterator, Tuple, Any


class Factory:
    def __init__(self, name: str):
        self.__name = name
        self.__bank = {}

    def register(self, name: str, hook: callable = None) -> callable:
        def decorator(cls):
            nonlocal name, hook
            return self.register_item(name=name, cls=cls, hook=hook)

        return decorator

    def register_item(self, name: str, cls: Any, hook: callable = None) -> Any:
        if name in self.__bank:
            raise RuntimeError(f"{name} already exists in the factory '{self.__name}'")
        if hook is not None:
            cls = hook(cls)
        self.__bank[name] = cls
        return cls

    def has(self, *names: str) -> bool:
        for name in names:
            if name not in self.__bank:
                return False
        return True

    def keys(self) -> List[str]:
        return self.__bank.keys()

    def get(self, name: str) -> Any:
        assert name
        assert (
            name in self.__bank
        ), f"{name} is not registered in the factory '{self.__name}'"
        return self.__bank.get(name)

    def __getitem__(self, name: str) -> Any:
        return self.get(name)

    def iter_unique(self) -> Iterator[Tuple[str, Any]]:
        already_sent = []
        for name, fct in self.__bank.items():
            if fct in already_sent:
                continue
            yield name, fct
            already_sent.append(fct)

    def items(self):
        return self.__bank.items()

    def __repr__(self):
        return f"Factory<{self.__name}>[{len(self.__bank)}]"


__FACTORY_BANK = Factory("factories")


def get_factory_bank() -> Factory:
    return __FACTORY_BANK


def get_factory(name: str) -> Factory:
    return get_factory_bank().get(name)

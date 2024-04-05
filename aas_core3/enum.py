"""Wrap an enumeration class in Micropython to follow the protocol of ``enum.Enum``."""


class Literal:
    def __init__(self, name, value):
        self.name = name
        self.value = value


def decorator(cls):
    for key, value in cls.__dict__.items():
        if key.startswith("__"):
            continue

        setattr(cls, key, Literal(key, value))

    return cls

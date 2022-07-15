"""Bulding objects from specs."""
import copy
from collections import defaultdict

REGISTRY: dict = defaultdict(dict)


def register(name: str, versions=None):
    """Decorator for registering classes to be buildable through a spec."""
    if versions is not None:
        raise NotImplementedError()

    def register_fn(cls):
        REGISTRY[name] = cls
        cls.spec_name = name
        return cls

    return register_fn


def get_cls_from_name(name: str):
    """Translates a string containing a type name used in a spec to the
    class registered to that name."""
    return REGISTRY[name]


def build(spec: dict):
    """Builds an object from the given spec."""
    cls = get_cls_from_name(spec["type"])

    spec_ = copy.copy(spec)
    del spec_["type"]
    obj = cls(**spec_)
    return obj

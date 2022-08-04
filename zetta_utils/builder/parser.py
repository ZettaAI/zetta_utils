"""Bulding objects from nested specs."""
import copy
from typing import Any, Callable
from typeguard import typechecked


REGISTRY: dict = {}
PARSE_KEY = "<type>"
RECURSE_KEY = "<recursive_parse>"


@typechecked
def register(name: str, versions=None) -> Callable:
    """Decorator for registering classes to be buildable.

    :param name: Name which will be used for to indicate an object of the
        decorated type.

    """
    if versions is not None:
        raise NotImplementedError()  # pragma: no cover

    def register_fn(cls):
        REGISTRY[name] = cls
        return cls

    return register_fn


@typechecked
def get_cls_from_name(name: str) -> Any:
    """Translate a string to the class registered with that name.

    :param name: Name to be translated.
    :return: Corresponding class.

    """
    return REGISTRY[name]


@typechecked
def build(spec: dict, must_build=True) -> Any:
    """Build an object from the given spec.

    :param spec: Input dictionary.
    :param must_build: Whether to throw a ``ValueError`` when the dictionary
        does not contain the ``<type>`` key in the outermost level.
    :return: Object build according to the specification.

    """
    if PARSE_KEY in spec:
        result = _build(spec)
    else:
        if must_build:
            raise ValueError(
                f"The spec to be parsed doesn't contain '{PARSE_KEY}' "
                "whille `must_build == True`."
            )
        result = spec

    return result


@typechecked
def _build(field: Any) -> Any:
    if isinstance(field, (bool, int, float, str)) or field is None:
        result = field  # type: Any
    elif isinstance(field, list):
        result = [_build(i) for i in field]
    elif isinstance(field, tuple):
        result = tuple(_build(i) for i in field)
    elif isinstance(field, dict):
        if PARSE_KEY in field:
            cls = get_cls_from_name(field[PARSE_KEY])
            field_ = copy.copy(field)
            del field_[PARSE_KEY]

            recurse = False
            if RECURSE_KEY not in field_ or field[RECURSE_KEY]:
                recurse = True

            if RECURSE_KEY in field_:
                del field_[RECURSE_KEY]

            if recurse:
                field_ = _build(field_)
            result = cls(**field_)
        else:
            result = {k: _build(v) for k, v in field.items()}
    else:
        raise ValueError(f"Unsupported type: {type(field)}")  # pragma: no cover

    return result

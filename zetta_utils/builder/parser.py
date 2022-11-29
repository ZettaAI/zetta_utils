"""Bulding objects from nested specs."""
from __future__ import annotations

import copy
from typing import Any, Callable, TypeVar

from typeguard import typechecked

from zetta_utils.partial import ComparablePartial

REGISTRY: dict = {}
PARSE_KEY = "@type"
MODE_KEY = "@mode"
RECURSE_KEY = "@recursive_parse"


T = TypeVar("T")


@typechecked
def register(name: str, versions=None) -> Callable[[T], T]:
    """Decorator for registering classes to be buildable.

    :param name: Name which will be used for to indicate an object of the
        decorated type.

    """
    if versions is not None:
        raise NotImplementedError(
            f"Versioning is not implemented. Specified versions: {versions}"
        )  # pragma: no cover

    def register_fn(cls: T) -> T:
        REGISTRY[name] = cls
        return cls

    return register_fn


@typechecked
def get_callable_from_name(name: str) -> Any:
    """Translate a string to the callable registered with that name.

    :param name: Name to be translated.
    :return: Corresponding class.

    """
    return REGISTRY[name]


@typechecked
def build(spec: dict, must_build=True) -> Any:
    """Build an object from the given spec.

    :param spec: Input dictionary.
    :param must_build: Whether to throw a ``ValueError`` when the dictionary
        does not contain the ``@type`` key in the outermost level.
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
def _build(field: Any) -> Any:  # pylint: disable=too-many-branches
    if isinstance(field, (bool, int, float, str)) or field is None:
        result = field  # type: Any
    elif isinstance(field, list):
        result = [_build(i) for i in field]
    elif isinstance(field, tuple):
        result = tuple(_build(i) for i in field)
    elif isinstance(field, dict):
        if PARSE_KEY in field:
            registered_fn = get_callable_from_name(field[PARSE_KEY])
            mode = field.get(MODE_KEY, "regular")

            recurse = False
            if RECURSE_KEY not in field or field[RECURSE_KEY]:
                recurse = True

            fn_kwargs = copy.copy(field)
            del fn_kwargs[PARSE_KEY]
            fn_kwargs.pop(MODE_KEY, None)  # deletes if exits
            fn_kwargs.pop(RECURSE_KEY, None)  # deletes if exits

            if recurse:
                fn_kwargs = _build(fn_kwargs)

            if mode == "regular":
                try:
                    result = registered_fn(**fn_kwargs)
                except Exception as e:  # pragma: no cover
                    if hasattr(registered_fn, "__name__"):
                        name = registered_fn.__name__
                    else:
                        name = str(registered_fn)
                    raise RuntimeError(
                        f'Exception while building "@type": "{field[PARSE_KEY]}" '
                        f"(mapped to '{name}' from module '{registered_fn.__module__}'), "
                        f'"@mode": "{mode}"'
                    ) from e
            elif mode == "partial":
                result = ComparablePartial(registered_fn, **fn_kwargs)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            # save the spec that was used to create the object if possible
            # slotted classes won't allow adding new attributes
            if hasattr(result, "__dict__"):
                object.__setattr__(result, "__init_builder_spec", field)

        else:
            result = {k: _build(v) for k, v in field.items()}
    else:
        raise ValueError(f"Unsupported type: {type(field)}")

    return result

"""Bulding objects from nested specs."""
from __future__ import annotations

import copy
from typing import Any, Callable, List, Optional, TypeVar, Union

from typeguard import typechecked

from zetta_utils.common.partial import ComparablePartial

from ..typing import IntVec3D, Vec3D

REGISTRY: dict = {}
REGISTRY_CAST_VEC3D: dict = {}
REGISTRY_CAST_INTVEC3D: dict = {}
PARSE_KEY = "@type"
MODE_KEY = "@mode"
RECURSE_KEY = "@recursive_parse"


T = TypeVar("T")


@typechecked
def register(
    name: str,
    cast_to_vec3d: Optional[List[str]] = None,
    cast_to_intvec3d: Optional[List[str]] = None,
    versions=None,
) -> Callable[[T], T]:
    """Decorator for registering classes to be buildable.

    :param name: Name which will be used for to indicate an object of the
        decorated type.

    """
    if versions is not None:
        raise NotImplementedError(
            f"Versioning is not implemented. Specified versions: {versions}"
        )  # pragma: no cover

    if cast_to_vec3d is None:
        cast_to_vec3d = []
    if cast_to_intvec3d is None:
        cast_to_intvec3d = []

    def register_fn(cls: T) -> T:
        REGISTRY[name] = cls
        REGISTRY_CAST_VEC3D[name] = cast_to_vec3d
        REGISTRY_CAST_INTVEC3D[name] = cast_to_intvec3d
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
def get_cast_to_vec3d_from_name(name: str) -> Any:
    """Translate a string to the name of arguments that are to be cast to Vec3D
    for the callable registered with that name.

    :param name: Name to be translated.
    :return: Corresponding names to be cast to Vec3D.

    """
    return REGISTRY_CAST_VEC3D[name]


@typechecked
def get_cast_to_intvec3d_from_name(name: str) -> Any:
    """Translate a string to the name of arguments that are to be cast to IntVec3D
    for the callable registered with that name.

    :param name: Name to be translated.
    :return: Corresponding names to be cast to IntVec3D.

    """
    return REGISTRY_CAST_INTVEC3D[name]


@typechecked
def build(spec: Union[dict, list], must_build: bool = False) -> Any:
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
            registered_cast_to_vec3d = get_cast_to_vec3d_from_name(field[PARSE_KEY])
            registered_cast_to_intvec3d = get_cast_to_intvec3d_from_name(field[PARSE_KEY])
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

            for kwarg_name in registered_cast_to_vec3d:
                if kwarg_name in fn_kwargs:
                    fn_kwargs[kwarg_name] = Vec3D(fn_kwargs[kwarg_name])

            for kwarg_name in registered_cast_to_intvec3d:
                if kwarg_name in fn_kwargs:
                    fn_kwargs[kwarg_name] = IntVec3D(fn_kwargs[kwarg_name])

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

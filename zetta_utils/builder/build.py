"""Bulding objects from nested specs."""
from __future__ import annotations

import json
from typing import Any, Callable, Final, Optional, Union

import attrs
from typeguard import typechecked

from zetta_utils import parsing
from zetta_utils.common import ctx_managers

from .registry import REGISTRY

SPECIAL_KEYS: Final = {
    "mode": "@mode",
    "type": "@type",
}


@typechecked
def build(
    spec: Optional[Union[dict, list]] = None,
    path: Optional[str] = None,
) -> Any:
    """Build an object from the given spec.

    :param spec: Input dictionary.
    :return: Object build according to the specification.

    """
    if spec is None and path is None or spec is not None and path is not None:
        raise ValueError("Exactly one of `spec`/`path` must be provided.")

    if spec is not None:
        final_spec = spec
    else:
        final_spec = parsing.cue.load(path)

    # error check the spec
    _traverse_spec(final_spec, _check_type_value, name_prefix="spec")

    # build the spec
    result = _traverse_spec(final_spec, _build_dict_spec, name_prefix="spec")

    return result


def _traverse_spec(spec: Any, apply_fn: Callable, name_prefix: str) -> Any:
    try:
        spec_as_str = json.dumps(spec)
    except TypeError:
        spec_as_str = '{"error": "Unserializable Spec"}'

    with ctx_managers.set_env_ctx_mngr(CURRENT_BUILD_SPEC=spec_as_str):
        if isinstance(spec, (bool, int, float, str)) or spec is None:
            result = spec  # type: Any
        elif isinstance(spec, list):
            result = [
                _traverse_spec(e, apply_fn, f"{name_prefix}[{i}]") for i, e in enumerate(spec)
            ]
        elif isinstance(spec, tuple):
            result = tuple(
                _traverse_spec(e, apply_fn, f"{name_prefix}[{i}]") for i, e in enumerate(spec)
            )
        elif isinstance(spec, dict):
            if SPECIAL_KEYS["type"] in spec:
                result = apply_fn(spec, name_prefix)
            else:
                result = {
                    k: _traverse_spec(v, apply_fn, f"{name_prefix}.{k}") for k, v in spec.items()
                }
        else:
            result = spec

    return result


def _check_type_value(spec: dict[str, Any], name_prefix: str) -> Any:
    if spec[SPECIAL_KEYS["type"]] not in REGISTRY:
        raise ValueError(
            f'Unregistered "{SPECIAL_KEYS["type"]}": "{spec[SPECIAL_KEYS["type"]]}" '
            f"for item {name_prefix}"
        )

    for k, v in spec.items():
        _traverse_spec(v, _check_type_value, f"{name_prefix}.{k}")


def _build_dict_spec(spec: dict[str, Any], name_prefix: str) -> Any:
    mode = spec.get(SPECIAL_KEYS["mode"], "regular")

    if mode == "regular":
        fn = REGISTRY[spec[SPECIAL_KEYS["type"]]].fn
        fn_kwargs = {
            k: _traverse_spec(v, _build_dict_spec, f"{name_prefix}.{k}")
            for k, v in spec.items()
            if k not in SPECIAL_KEYS.values()
        }

        try:
            result = fn(**fn_kwargs)
        except Exception as e:  # pragma: no cover
            if hasattr(fn, "__name__"):
                name = fn.__name__
            else:
                name = str(fn)
            e.args = (
                f'{e}\nException occured while building "{name_prefix}" '
                f'with "@type" "{spec[SPECIAL_KEYS["type"]]}" '
                f'(mapped to "{name}" from module "{fn.__module__}", "@mode": "{mode}")',
            )
            raise e from None
    elif mode == "partial":
        if not REGISTRY[spec[SPECIAL_KEYS["type"]]].allow_partial:
            raise ValueError(
                f'"@mode": "partial" is not allowed for '
                f'"@type": "{spec[SPECIAL_KEYS["type"]]}"'
            )
        result = BuilderPartial(spec=spec)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # save the spec that was used to create the object if possible
    # slotted classes won't allow adding new attributes
    if hasattr(result, "__dict__"):
        object.__setattr__(result, "__built_with_spec", spec)

    return result


@typechecked
@attrs.mutable(slots=False)
class BuilderPartial:
    """
    A partial function specification based on builder specs.
    Useful to efficiently serialize partially spec-ed constructs.
    """

    spec: dict[str, Any]
    _built_spec_kwargs: dict[str, Any] | None = attrs.field(init=False, default=None)

    def get_display_name(self):  # pragma: no cover # pretty print
        if SPECIAL_KEYS["type"] in self.spec:
            return self.spec[SPECIAL_KEYS["type"]]
        else:
            return "BuilderPartial"

    def _get_built_spec_kwargs(self) -> dict[str, Any]:
        if self._built_spec_kwargs is None:
            self._built_spec_kwargs = {
                k: _traverse_spec(v, _build_dict_spec, f"partial.{k}")
                for k, v in self.spec.items()
                if k not in SPECIAL_KEYS.values()
            }
        return self._built_spec_kwargs

    def __call__(self, *args, **kwargs):
        try:
            spec_as_str = json.dumps({**self.spec, **kwargs, "__args": args})
        except TypeError:
            spec_as_str = '{"error": "Unserializable Spec"}'

        with ctx_managers.set_env_ctx_mngr(CURRENT_BUILD_SPEC=spec_as_str):
            fn = REGISTRY[self.spec[SPECIAL_KEYS["type"]]].fn
            result = fn(
                *args,
                **self._get_built_spec_kwargs(),
                **kwargs,
            )
            return result

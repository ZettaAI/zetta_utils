"""Bulding objects from nested specs."""
from __future__ import annotations

from typing import Any, Callable, Final, Optional

import attrs
from typeguard import typechecked

from zetta_utils import parsing
from zetta_utils.common import ctx_managers
from zetta_utils.parsing import json
from zetta_utils.typing import JsonDict

from . import constants
from .registry import get_matching_entry

SPECIAL_KEYS: Final = {
    "mode": "@mode",
    "type": "@type",
    "version": "@version",
}

BUILT_OBJECT_ID_REGISTRY: dict[int, JsonDict] = {}


def get_initial_builder_spec(obj: Any) -> JsonDict | None:
    """Returns the builder spec that the object was initially built with.
    Note that mutations to the object after it was built will not be
    reflected in the spec. Returns `None` if the object was not built with
    builder
    """
    # breakpoint()
    result = BUILT_OBJECT_ID_REGISTRY.get(id(obj), None)
    return result


@typechecked
def build(
    spec: dict | list | None = None,
    path: str | None = None,
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
    _traverse_spec(
        final_spec,
        _check_type_value,
        name_prefix="spec",
        version=constants.DEFAULT_VERSION,
    )

    # build the spec
    result = _traverse_spec(
        final_spec,
        _build_dict_spec,
        name_prefix="spec",
        version=constants.DEFAULT_VERSION,
    )

    return result


def _traverse_spec(spec: Any, apply_fn: Callable, name_prefix: str, version: str) -> Any:
    try:
        spec_as_str = json.dumps(spec)
    except TypeError:
        spec_as_str = '{"error": "Unserializable Spec"}'

    with ctx_managers.set_env_ctx_mngr(CURRENT_BUILD_SPEC=spec_as_str):
        if isinstance(spec, (bool, int, float, str)) or spec is None:
            result = spec  # type: Any
        elif isinstance(spec, list):
            result = [
                _traverse_spec(
                    spec=e,
                    apply_fn=apply_fn,
                    name_prefix=f"{name_prefix}[{i}]",
                    version=version,
                )
                for i, e in enumerate(spec)
            ]
        elif isinstance(spec, tuple):
            result = tuple(
                _traverse_spec(
                    spec=e,
                    apply_fn=apply_fn,
                    name_prefix=f"{name_prefix}[{i}]",
                    version=version,
                )
                for i, e in enumerate(spec)
            )
        elif isinstance(spec, dict):
            if SPECIAL_KEYS["type"] in spec:
                result = apply_fn(spec, name_prefix, version=version)
            else:
                result = {
                    k: _traverse_spec(
                        spec=v,
                        apply_fn=apply_fn,
                        name_prefix=f"{name_prefix}.{k}",
                        version=version,
                    )
                    for k, v in spec.items()
                }
        else:
            result = spec
    BUILT_OBJECT_ID_REGISTRY[id(result)] = spec
    return result


def _check_type_value(spec: dict[str, Any], name_prefix: str, version: str) -> Any:
    if SPECIAL_KEYS["version"] in spec:
        version = spec[SPECIAL_KEYS["version"]]

    this_type = spec[SPECIAL_KEYS["type"]]

    get_matching_entry(this_type, version=version)
    for k, v in spec.items():
        _traverse_spec(
            v,
            apply_fn=_check_type_value,
            name_prefix=f"{name_prefix}.{k}",
            version=version,
        )


def _build_dict_spec(spec: dict[str, Any], name_prefix: str, version: str) -> Any:
    this_mode = spec.get(SPECIAL_KEYS["mode"], "regular")
    this_type = spec[SPECIAL_KEYS["type"]]

    if SPECIAL_KEYS["version"] in spec:
        version = spec[SPECIAL_KEYS["version"]]

    if this_mode == "regular":
        fn = get_matching_entry(this_type, version=version).fn
        fn_kwargs = {
            k: _traverse_spec(
                v,
                apply_fn=_build_dict_spec,
                name_prefix=f"{name_prefix}.{k}",
                version=version,
            )
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
                f'with "@type" "{this_type}" '
                f'(mapped to "{name}" from module "{fn.__module__}", "@mode": "{this_mode}")',
            )
            raise e from None
    elif this_mode == "partial":
        if not get_matching_entry(this_type, version=version).allow_partial:
            raise ValueError(
                f'"@mode": "partial" is not allowed for '
                f'"@type": "{spec[SPECIAL_KEYS["type"]]}"'
            )
        result = BuilderPartial(spec=spec)
    else:
        raise ValueError(f"Unsupported mode: {this_mode}")

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
    name: Optional[str] = None

    def get_display_name(self):  # pragma: no cover # pretty print
        if self.name is not None:
            return self.name
        elif SPECIAL_KEYS["type"] in self.spec:
            return self.spec[SPECIAL_KEYS["type"]]
        else:
            return "BuilderPartial"

    def _get_built_spec_kwargs(self, version: str) -> dict[str, Any]:
        if self._built_spec_kwargs is None:
            self._built_spec_kwargs = {
                k: _traverse_spec(
                    v,
                    apply_fn=_build_dict_spec,
                    name_prefix=f"partial.{k}",
                    version=version,
                )
                for k, v in self.spec.items()
                if k not in SPECIAL_KEYS.values()
            }
        return self._built_spec_kwargs

    def __call__(self, *args, **kwargs):
        try:
            spec_as_str = json.dumps({**self.spec, **kwargs, "__args": args})
        except TypeError:  # pragma: no cover
            spec_as_str = '{"error": "Unserializable Spec"}'

        with ctx_managers.set_env_ctx_mngr(CURRENT_BUILD_SPEC=spec_as_str):
            version = self.spec.get(SPECIAL_KEYS["version"], constants.DEFAULT_VERSION)

            fn = get_matching_entry(self.spec[SPECIAL_KEYS["type"]], version=version).fn
            result = fn(
                *args,
                **self._get_built_spec_kwargs(version=version),
                **kwargs,
            )
            return result

"""Bulding objects from nested specs."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Final, Optional, cast

import attrs
import cachetools
from typeguard import typechecked

from zetta_utils import parsing
from zetta_utils.common import ctx_managers
from zetta_utils.parsing import json
from zetta_utils.typing import JsonDict, JsonSerializableValue

from . import constants
from .registry import get_matching_entry

SPECIAL_KEYS: Final = {
    "mode": "@mode",
    "type": "@type",
    "version": "@version",
}
BUILT_OBJECT_ID_REGISTRY: dict[int, JsonSerializableValue] = {}


def get_initial_builder_spec(obj: Any) -> JsonSerializableValue:
    """Returns the builder spec that the object was initially built with.
    Note that mutations to the object after it was built will not be
    reflected in the spec. Returns `None` if the object was not built with
    builder
    """
    result = BUILT_OBJECT_ID_REGISTRY.get(id(obj), None)
    return result


@typechecked
def build(spec: dict | list | None = None, path: str | None = None, parallel: bool = False) -> Any:
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

    result = _build(
        spec=final_spec, parallel=parallel, name_prefix="spec", version=constants.DEFAULT_VERSION
    )

    return result


def _build(spec: JsonSerializableValue, parallel: bool, version: str, name_prefix: str) -> Any:
    stages = _parse_stages(spec, version=version, name_prefix=name_prefix)
    result = _execute_build_stages(stages=stages, parallel=parallel)
    return result


class UnpicklableDict(dict):
    def __reduce__(self):
        """Ensure that the object becomes an empty dictionary when pickled."""
        return (UnpicklableDict, (), None, None, iter({}.items()))


@cachetools.cached(UnpicklableDict())
def _get_process_pool():
    return ProcessPoolExecutor()


@attrs.mutable
class ObjectToBeBuilt:
    spec: JsonSerializableValue
    fn: Callable
    name_prefix: str
    kwargs: dict[str, Any] = attrs.field(factory=dict)
    parent: ObjectToBeBuilt | None = None
    parent_kwarg_name: str | None = None
    allow_parallel: bool = True


def _build_object(
    fn: Callable, kwargs: dict[str, Any], spec: JsonSerializableValue, name_prefix: str
) -> Any:
    spec_as_str = json.dumps(spec)
    with ctx_managers.set_env_ctx_mngr(CURRENT_BUILD_SPEC=spec_as_str):
        try:
            result = fn(**kwargs)
        except Exception as e:  # pragma: no cover
            if hasattr(fn, "__name__"):
                name = fn.__name__
            else:
                name = str(fn)
            e.args = (
                f'{e}\nException occured while building "{name_prefix}" '
                f'(mapped to "{name}" from module "{fn.__module__}")',
            )
            raise e from None
    return result


@attrs.mutable
class Stage:
    sequential_part: list[ObjectToBeBuilt] = attrs.field(factory=list)
    parallel_part: list[ObjectToBeBuilt] = attrs.field(factory=list)


def _execute_build_stages(stages: list[Stage], parallel: bool):
    assert len(stages) > 0

    def _process_result(obj: ObjectToBeBuilt, result: Any):
        # for obj, result in zip(objs, results):
        BUILT_OBJECT_ID_REGISTRY[id(result)] = obj.spec
        if obj.parent is not None:
            assert obj.parent_kwarg_name is not None
            obj.parent.kwargs[obj.parent_kwarg_name] = result

    for stage in stages:
        for obj in stage.sequential_part:
            obj_result = _build_object(
                fn=obj.fn,
                kwargs=obj.kwargs,
                spec=obj.spec,
                name_prefix=obj.name_prefix,
            )
            _process_result(obj, obj_result)

        if parallel:
            futures = [
                _get_process_pool().submit(
                    _build_object,
                    obj.fn,
                    obj.kwargs,
                    obj.spec,
                    obj.name_prefix,
                )
                for obj in stage.parallel_part
            ]
            results_parallel = [fut.result() for fut in futures]
            for obj, result in zip(stage.parallel_part, results_parallel):
                _process_result(obj, result)
            # the last parallel built object is returned if the
            # last stage has parallel part
            if len(results_parallel) > 0:
                obj_result = results_parallel[-1]
        else:
            for obj in stage.parallel_part:
                obj_result = _build_object(
                    fn=obj.fn,
                    kwargs=obj.kwargs,
                    spec=obj.spec,
                    name_prefix=obj.name_prefix,
                )
                _process_result(obj, obj_result)

    return obj_result


def _build_list(**kwargs):
    return list(kwargs.values())


def _build_tuple(**kwargs):  # pragma: no cover
    return tuple(kwargs.values())


def _build_dict(**kwargs):
    return kwargs


def _is_trivial(obj) -> bool:
    return (
        isinstance(obj, (int, float, bool, str, dict, list, tuple, BuilderPartial)) or obj is None
    )


def _parse_stages(spec: JsonSerializableValue, name_prefix: str, version: str) -> list[Stage]:
    stage_dict: dict[int, Stage] = defaultdict(Stage)
    final_obj = _parse_stages_inner(
        spec=spec,
        stages_dict=stage_dict,
        level=0,
        version=version,
        parent=None,
        parent_kwarg_name=None,
        name_prefix=name_prefix,
    )
    result = [stage_dict[k] for k in reversed(sorted(stage_dict.keys()))]
    if _is_trivial(final_obj):
        result.append(
            Stage(
                sequential_part=[
                    ObjectToBeBuilt(spec=spec, fn=lambda: final_obj, name_prefix="spec")
                ]
            )
        )
    return result


def _parse_stages_inner(  # pylint: disable=too-many-branches,too-many-statements
    spec: JsonSerializableValue,
    stages_dict: dict[int, Stage],
    level: int,
    version: str,
    name_prefix: str,
    parent: ObjectToBeBuilt | None,
    parent_kwarg_name: str | None,
) -> ObjectToBeBuilt | JsonSerializableValue | BuilderPartial:
    result: ObjectToBeBuilt | JsonSerializableValue | BuilderPartial
    if isinstance(spec, (int, float, bool, str)) or spec is None:
        result = spec
    elif isinstance(spec, (list, tuple)):
        this_obj = ObjectToBeBuilt(
            spec=spec,
            fn=_build_list if isinstance(spec, list) else _build_tuple,
            parent=parent,
            parent_kwarg_name=parent_kwarg_name,
            name_prefix=name_prefix,
        )

        args = [
            _parse_stages_inner(
                e,
                stages_dict=stages_dict,
                level=level,
                parent=this_obj,
                parent_kwarg_name=str(i),
                version=version,
                name_prefix=f"{name_prefix}[{i}]",
            )
            for i, e in enumerate(spec)
        ]

        if all(_is_trivial(e) for e in args):
            result = type(spec)(cast(list[JsonSerializableValue], args))
        else:
            if any(not v.allow_parallel for v in args if isinstance(v, ObjectToBeBuilt)):
                this_obj.allow_parallel = False
            this_obj.kwargs = {str(i): v for i, v in enumerate(args) if _is_trivial(v)}
            stages_dict[level].sequential_part.append(this_obj)
            result = this_obj

    elif isinstance(spec, dict) and "@type" not in spec:
        assert "@mode" not in spec
        this_obj = ObjectToBeBuilt(
            spec=spec,
            fn=_build_dict,
            parent=parent,
            parent_kwarg_name=parent_kwarg_name,
            name_prefix=name_prefix,
        )
        kwargs = {
            k: _parse_stages_inner(
                v,
                stages_dict=stages_dict,
                level=level,
                parent=this_obj,
                parent_kwarg_name=k,
                version=version,
                name_prefix=f"{name_prefix}[{k}]",
            )
            for k, v in spec.items()
        }
        all_basic = all(_is_trivial(e) for e in kwargs.values())
        if all_basic:
            result = cast(JsonSerializableValue, kwargs)
        else:
            if any(
                not v.allow_parallel for v in kwargs.values() if isinstance(v, ObjectToBeBuilt)
            ):
                this_obj.allow_parallel = False
            this_obj.kwargs = {k: v for k, v in kwargs.items() if _is_trivial(v)}
            stages_dict[level].sequential_part.append(this_obj)
            result = this_obj
    elif isinstance(spec, dict):
        assert "@type" in spec

        if "@version" in spec:
            version = cast(str, spec["@version"])
        this_type = spec["@type"]
        assert isinstance(this_type, str)
        entry = get_matching_entry(this_type, version)
        if "@mode" in spec and spec["@mode"] == "partial":
            if not entry.allow_partial:
                raise ValueError(f'`"@mode": partial` not allowed for `"@type": {this_type}"`')
            assert isinstance(spec, dict)
            result = BuilderPartial(spec={k: v for k, v in spec.items() if k != "@mode"})
        else:
            if "@mode" in spec and spec["@mode"] != "regular":
                raise ValueError(f"Unsupported mode {spec['@mode']}")
            this_obj = ObjectToBeBuilt(
                spec=spec,
                fn=entry.fn,
                parent=parent,
                parent_kwarg_name=parent_kwarg_name,
                name_prefix=name_prefix,
                allow_parallel=entry.allow_parallel,
            )
            kwargs = {
                k: _parse_stages_inner(
                    v,
                    stages_dict=stages_dict,
                    level=level + 1,
                    parent=this_obj,
                    parent_kwarg_name=k,
                    version=version,
                    name_prefix=f"{name_prefix}.{k}",
                )
                for k, v in spec.items()
                if not k.startswith("@")
            }
            this_obj.kwargs = {k: v for k, v in kwargs.items() if _is_trivial(v)}
            if any(
                not v.allow_parallel for v in kwargs.values() if isinstance(v, ObjectToBeBuilt)
            ):
                this_obj.allow_parallel = False
            if entry.allow_parallel:
                stages_dict[level + 1].parallel_part.append(this_obj)
            else:
                stages_dict[level + 1].sequential_part.append(this_obj)
            result = this_obj
    else:
        raise ValueError(f"Unsupported type: {type(spec)}")

    return result


@typechecked
@attrs.mutable(slots=False)
class BuilderPartial:
    """
    A partial function specification based on builder specs.
    Useful to efficiently serialize partially spec-ed constructs.
    """

    spec: JsonDict
    _built_spec_kwargs: dict[str, JsonSerializableValue] | None = attrs.field(
        init=False, default=None
    )
    name: Optional[str] = None
    parallel: bool = False

    def get_display_name(self):  # pragma: no cover # pretty print
        if self.name is not None:
            return self.name
        elif isinstance(self.spec, dict) and SPECIAL_KEYS["type"] in self.spec:
            return self.spec[SPECIAL_KEYS["type"]]
        else:
            return "BuilderPartial"

    def _get_built_spec_kwargs(self, version: str) -> dict[str, Any]:
        if self._built_spec_kwargs is None:
            self._built_spec_kwargs = {
                k: _build(
                    spec=v,
                    name_prefix=f"partial.{k}",
                    version=version,
                    parallel=self.parallel,
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
            version = cast(str, self.spec.get(SPECIAL_KEYS["version"], constants.DEFAULT_VERSION))

            fn = get_matching_entry(cast(str, self.spec[SPECIAL_KEYS["type"]]), version=version).fn
            result = fn(
                *args,
                **self._get_built_spec_kwargs(version=version),
                **kwargs,
            )
            return result

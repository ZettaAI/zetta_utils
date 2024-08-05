from __future__ import annotations

import functools
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Type,
    Union,
    runtime_checkable,
)

import attrs
from typing_extensions import ParamSpec

from . import id_generation
from .tasks import Task

BatchType = Optional[Union["Dependency", List[Task], List["Flow"]]]
FlowFnYieldType = Union["Dependency", "Task", List["Task"], "Flow", List["Flow"]]
FlowFnReturnType = Generator[FlowFnYieldType, None, Any]

P = ParamSpec("P")


@attrs.mutable
class Dependency:
    wait_for: Optional[Union[Union["Flow", "Task"], List[Union["Flow", "Task"]]]] = None
    ids: Optional[Set[str]] = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        if self.wait_for is None:
            self.ids = None
        elif isinstance(self.wait_for, list):
            self.ids = set(e.id_ for e in self.wait_for)
        else:
            self.ids = set([self.wait_for.id_])


@runtime_checkable
class FlowSchema(Protocol[P]):
    """
    Interface of a flow Schema -- a callable that returns a mazepa Flow.
    """

    def __call__(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Flow:
        ...


@runtime_checkable
class RawFlowSchemaCls(Protocol[P]):
    """
    Interface for a type that can be decorated with ``@flow_schema_cls``.
    """

    def flow(self, *args: P.args, **kwargs: P.kwargs) -> FlowFnReturnType:
        ...


@attrs.mutable
class Flow:
    """
    Implementation of mazepa flow.
    Users are expected to use ``flow_schema`` and ``flow_schema_cls`` decorators rather
    than using this class directly.
    """

    fn: Callable[..., FlowFnReturnType]
    id_: str = attrs.field(factory=lambda: str(uuid.uuid1()))
    _iterator: FlowFnReturnType = attrs.field(init=False, default=None)
    tags: list[str] = attrs.field(factory=list)

    # These are saved as attributes just for printability.
    args: Iterable = attrs.field(factory=list)
    kwargs: Dict = attrs.field(factory=dict)
    _has_been_called: bool = attrs.field(init=False, default=False)

    def add_tags(self, tags: list[str]) -> Flow:  # pragma: no cover
        self.tags += tags
        return self

    def _set_up(
        self,
        *args: Iterable,
        **kwargs: Dict,
    ):
        self.args = args
        self.kwargs = kwargs

    def get_next_batch(self) -> BatchType:
        if not self._has_been_called:
            self._iterator = self.fn(*self.args, **self.kwargs)
            self._has_been_called = True

        yielded = next(self._iterator, None)

        result: BatchType
        if isinstance(yielded, Flow):
            result = [yielded]
        elif isinstance(yielded, Task):
            result = [yielded]
        else:
            result = yielded

        if self.tags is not None and isinstance(result, list):
            for e in result:
                e.tags += self.tags

        return result


@attrs.mutable
class _FlowSchema(Generic[P]):
    """
    Wrapper that makes a FlowSchema from a callable.
    Users are expected to use ``@flow`` and ``@flow_cls`` decorators rather
    than using this class directly.
    """

    fn: Callable[P, FlowFnReturnType]
    id_fn: Callable[[Callable, list, dict], str] = attrs.field(
        default=functools.partial(id_generation.generate_invocation_id, prefix="flow")
    )
    tags: list[str] = attrs.field(factory=list)

    def __call__(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Flow:
        id_ = self.id_fn(self.fn, list(args), kwargs)
        result = Flow(
            fn=self.fn,
            id_=id_,
            tags=self.tags,
        )
        result._set_up(*args, **kwargs)  # pylint: disable=protected-access # friend class
        return result


def flow_schema(fn: Callable[P, FlowFnReturnType]) -> FlowSchema[P]:
    """Decorator for generator functions defining mazepa flows."""
    return _FlowSchema[P](fn)


def flow_schema_cls(cls: Type[RawFlowSchemaCls]):
    # original_call = cls.__call__
    # TODO: figure out how to handle this with changing TaskExecutionEnvs
    def _call_fn(self, *args, **kwargs):
        return _FlowSchema(
            self.flow,
            # functools.partial(original_call, self),
            # TODO: Other params passed to decorator
        )(
            *args, **kwargs
        )  # pylint: disable=protected-access

    # can't override __new__ because of interaction with attrs/dataclass
    setattr(cls, "__call__", _call_fn)
    return cls


@flow_schema
def concurrent_flow(stages: Sequence[Flow | Task]):
    yield from stages


@flow_schema
def sequential_flow(stages: Sequence[Flow | Task]):
    for e in stages:
        yield e
        yield Dependency()

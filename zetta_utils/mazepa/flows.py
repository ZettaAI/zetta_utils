from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Type,
    Union,
    runtime_checkable,
)

import attrs
from typing_extensions import ParamSpec

from . import id_generation
from .dependency import Dependency
from .task_execution_env import TaskExecutionEnv
from .tasks import Task

BatchType = Optional[Union[Dependency, List[Task], List["Flow"]]]
FlowFnYieldType = Union[Dependency, "Task", List["Task"], "Flow", List["Flow"]]
FlowFnReturnType = Generator[FlowFnYieldType, None, Any]

P = ParamSpec("P")


@runtime_checkable
class Flow(Protocol):
    """
    Interface for a mazepa Flow.
    """

    fn: Callable[..., FlowFnReturnType]
    id_: str
    task_execution_env: Optional[TaskExecutionEnv]
    _iterator: FlowFnReturnType
    args: Iterable
    kwargs: Dict

    @contextmanager
    def task_execution_env_ctx(self, env: Optional[TaskExecutionEnv]) -> Iterator:
        ...

    def get_next_batch(self) -> BatchType:
        ...


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
class _Flow:
    """
    Implementation of mazepa flow.
    Users are expected to use ``flow`` and ``flow_cls`` decorators rather
    than using this class directly.
    """

    fn: Callable[..., FlowFnReturnType]
    id_: str
    task_execution_env: Optional[TaskExecutionEnv]
    _iterator: FlowFnReturnType = attrs.field(init=False, default=None)

    # These are saved as attributes just for printability.
    args: Iterable = attrs.field(init=False, default=list)
    kwargs: Dict = attrs.field(init=False, factory=dict)

    def _set_up(
        self,
        *args: Iterable,
        **kwargs: Dict,
    ):
        self.args = args
        self.kwargs = kwargs
        self._iterator = self.fn(*args, **kwargs)

    @contextmanager
    def task_execution_env_ctx(self, env: Optional[TaskExecutionEnv]):
        old_env = self.task_execution_env
        self.task_execution_env = env
        yield
        self.task_execution_env = old_env

    def get_next_batch(self) -> BatchType:
        yielded = next(self._iterator, None)

        result: BatchType
        if isinstance(yielded, Flow):
            result = [yielded]
        elif isinstance(yielded, Task):
            result = [yielded]
        else:
            result = yielded

        if self.task_execution_env is not None and isinstance(result, list):
            for e in result:
                e.task_execution_env = self.task_execution_env

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
    task_execution_env: TaskExecutionEnv = attrs.field(factory=TaskExecutionEnv)

    def __call__(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Flow:
        id_ = self.id_fn(self.fn, list(args), kwargs)
        result = _Flow(
            fn=self.fn,
            id_=id_,
            task_execution_env=self.task_execution_env,
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

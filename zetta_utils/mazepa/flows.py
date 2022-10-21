from __future__ import annotations

from typing import (
    Callable,
    Generator,
    Union,
    Type,
    Any,
    Optional,
    List,
    Generic,
    Iterable,
    Dict,
    Protocol,
    runtime_checkable,
)
from contextlib import contextmanager
from typing_extensions import ParamSpec
import attrs
from . import id_generators
from .tasks import Task
from .dependency import Dependency
from .task_execution_env import TaskExecutionEnv


BatchType = Optional[Union[Dependency, List[Task], List["Flow"]]]
FlowFnYieldType = Union[Dependency, "Task", List["Task"], "Flow", List["Flow"]]
FlowFnReturnType = Generator[FlowFnYieldType, None, Any]
P = ParamSpec("P")
P1 = ParamSpec("P1")


@runtime_checkable
class Flow(Protocol[P]):  # pragma: no cover # protocol
    """
    Wraps a callable generator to convert it into a mazepa flow.
    """

    fn: Callable[P, FlowFnReturnType]
    id_: str
    task_execution_env: Optional[TaskExecutionEnv]
    _iterator: FlowFnReturnType
    args: Iterable
    kwargs: Dict

    def _set_up(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        ...

    # @contextmanager
    def task_execution_env_ctx(self, env: Optional[TaskExecutionEnv]):
        ...

    def get_next_batch(self) -> BatchType:
        ...


@attrs.mutable
class _Flow(Generic[P]):
    """
    Implementation of mazepa flow.
    Users are expected to use ``flow`` and ``flow_cls`` decorators rather
    than using this class directly.
    """

    fn: Callable[P, FlowFnReturnType]
    id_: str
    task_execution_env: Optional[TaskExecutionEnv]
    _iterator: FlowFnReturnType = attrs.field(init=False, default=None)

    # These are saved as attributes just for printability.
    args: Iterable = attrs.field(init=False, default=list)
    kwargs: Dict = attrs.field(init=False, factory=dict)

    def _set_up(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
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


@runtime_checkable
class FlowType(Protocol[P]):  # pragma: no cover # protocol
    """
    Wraps generator callable to return a mazepa Flow when called.
    """

    def __call__(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Flow[P]:
        ...


@attrs.mutable
class _FlowType(Generic[P]):
    """
    Implementation of FlowType.
    Users are expected to use ``flow`` and ``flow_cls`` decorators rather
    than using this class directly.
    """

    fn: Callable[P, FlowFnReturnType]
    id_fn: Callable[[Callable, dict], str] = attrs.field(default=id_generators.get_unique_id)
    task_execution_env: TaskExecutionEnv = attrs.field(factory=TaskExecutionEnv)

    def __call__(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Flow[P]:  # pragma: no cover
        id_ = self.id_fn(self.fn, kwargs)
        result = _Flow[P](
            fn=self.fn,
            id_=id_,
            task_execution_env=self.task_execution_env,
        )
        result._set_up(*args, **kwargs)  # pylint: disable=protected-access # friend class
        return result


def flow_type(fn: Callable[P, FlowFnReturnType]) -> FlowType[P]:
    """Decorator for generator functions defining mazepa flows."""
    return _FlowType[P](fn)


# TODO: make static type checking detect when a class wihtout `generate` is decorated.
# class FlowGenerator(Protocol):
# generate: Callable[..., FlowFnReturnType]

# def generate(self, **kwargs) -> FlowFnReturnType: # pylint: disable=no-method-argument
# def generate(self, *args: P.args, **kwargs: P.kwargs) -> FlowFnReturnType: # pylint: disable=no-method-argument


def flow_type_cls(cls: Type) -> Type:  # rely on mypy plugin to do this properly
    # original_call = cls.__call__

    # TODO: figure out how to handle this with changing TaskExecutionEnvs
    def _call_fn(self, *args, **kwargs):
        return _FlowType(
            self.generate,
            # functools.partial(original_call, self),
            # TODO: Other params passed to decorator
        )(
            *args, **kwargs
        )  # pylint: disable=protected-access

    # can't override __new__ because it doesn't combine well with attrs/dataclass
    setattr(cls, "__call__", _call_fn)
    return cls

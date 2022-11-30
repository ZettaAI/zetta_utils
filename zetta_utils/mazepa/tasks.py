from __future__ import annotations

import functools
import time
import uuid
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    Protocol,
    Type,
    TypeVar,
    runtime_checkable,
)

import attrs
from typing_extensions import ParamSpec

from zetta_utils import log

from . import ctx_vars, id_generation
from .task_execution_env import TaskExecutionEnv
from .task_outcome import TaskOutcome, TaskStatus

logger = log.get_logger("mazepa")

R_co = TypeVar("R_co", covariant=True)
P = ParamSpec("P")


@attrs.mutable
class Task(Generic[R_co]):
    """
    An executable task.
    """

    fn: Callable[..., R_co]
    id_: str = attrs.field(factory=lambda: str(uuid.uuid1()))
    task_execution_env: TaskExecutionEnv = attrs.field(factory=TaskExecutionEnv)
    args_are_set: bool = attrs.field(init=False, default=False)
    args: Iterable = attrs.field(init=False, factory=list)
    kwargs: Dict = attrs.field(init=False, factory=dict)

    _mazepa_callbacks: list[Callable] = attrs.field(factory=list)
    outcome: TaskOutcome = attrs.field(
        factory=functools.partial(
            TaskOutcome,
            status=TaskStatus.NOT_SUBMITTED,
        )
    )
    # cache_expiration: datetime.timedelta = None
    # max_retry: # Can use SQS approximateReceiveCount to explicitly fail the task

    # Split into __init__ and _set_up because ParamSpec doesn't allow us
    # to play with kwargs.
    # cc: https://peps.python.org/pep-0612/#concatenating-keyword-parameters
    def _set_up(self, *args: Iterable, **kwargs: Dict):
        assert not self.args_are_set
        self.args = args
        self.kwargs = kwargs
        self.args_are_set = True

    def __call__(self) -> TaskOutcome[R_co]:
        assert self.args_are_set
        ctx_vars.task_id.set(self.id_)
        logger.debug(f"STARTING: Execution of {self}.")
        time_start = time.time()
        try:
            # TODO: parametrize by task execution environment
            return_value = self.fn(*self.args, **self.kwargs)
            status = TaskStatus.SUCCEEDED
            exception = None
        # Todo: catch special exceptions
        except Exception as exc:  # pylint: disable=broad-except
            exception = exc
            return_value = None
            status = TaskStatus.FAILED

        time_end = time.time()

        self.outcome = TaskOutcome(
            status=status,
            exception=exception,
            execution_secs=time_end - time_start,
            return_value=return_value,
        )
        for callback in self._mazepa_callbacks:
            callback(task=self)

        logger.debug(f"DONE: Execution of {self}.")
        ctx_vars.task_id.set(None)

        return self.outcome


@runtime_checkable
class TaskableOperation(Protocol[P, R_co]):
    """
    Wraps a callable to add a ``make_task`` method,
    ``make_task`` method creates a mazepa task corresponding to execution of
    the callable with the given parameters.
    """

    def __call__(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R_co:
        ...

    def make_task(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Task[R_co]:
        ...


@runtime_checkable
class RawTaskableOperationCls(Protocol[P, R_co]):
    """
    Interface of a class that can be wrapped by `@taskable_operation_cls`.
    """

    def __call__(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R_co:
        ...


@attrs.mutable
class _TaskableOperation(Generic[P, R_co]):
    """
    TaskableOperation wrapper
    """

    fn: Callable[P, R_co]
    id_fn: Callable[[Callable, list, dict], str] = attrs.field(
        default=functools.partial(id_generation.generate_invocation_id, prefix="task")
    )
    task_execution_env: TaskExecutionEnv = attrs.field(factory=TaskExecutionEnv)
    # max_retry: # Even for SQS, can use approximateReceiveCount to explicitly fail the task

    def __call__(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R_co:  # pragma: no cover # no logic
        return self.fn(*args, **kwargs)

    def make_task(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Task[R_co]:
        id_ = self.id_fn(self.fn, list(args), kwargs)
        result = Task[R_co](fn=self.fn, id_=id_, task_execution_env=self.task_execution_env)
        result._set_up(*args, **kwargs)  # pylint: disable=protected-access # friend class
        return result


def taskable_operation(fn: Callable[P, R_co]) -> TaskableOperation[P, R_co]:
    return _TaskableOperation(fn=fn)


def taskable_operation_cls(cls: Type[RawTaskableOperationCls]):
    def _make_task(self, *args, **kwargs):
        return _TaskableOperation(  # pylint: disable=protected-access
            self,
            # TODO: Other params passed to decorator
        ).make_task(
            *args, **kwargs
        )  # pylint: disable=protected-access

    # can't override __new__ because it doesn't combine well with attrs/dataclass
    setattr(cls, "make_task", _make_task)
    return cls

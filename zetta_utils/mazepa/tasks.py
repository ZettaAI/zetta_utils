from __future__ import annotations

import functools
import sys
import threading
import time
import traceback
import uuid
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    Optional,
    Protocol,
    Type,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)

import attrs
from typing_extensions import ParamSpec

from zetta_utils import log

from . import exceptions, id_generation
from .task_outcome import TaskOutcome, TaskStatus

logger = log.get_logger("mazepa")

R_co = TypeVar("R_co", covariant=True)
P = ParamSpec("P")


class RepeatTimer(threading.Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


@attrs.mutable
class TaskUpkeepSettings:
    perform_upkeep: bool = False
    interval_secs: Optional[float] = None
    callbacks: list[Callable] = attrs.field(factory=list)


@attrs.mutable
class Task(Generic[R_co]):  # pylint: disable=too-many-instance-attributes
    """
    An executable task.
    """

    fn: Callable[..., R_co]
    operation_name: str = "Unclassified Task"
    id_: str = attrs.field(factory=lambda: str(uuid.uuid1()))
    tags: list[str] = attrs.field(factory=list)

    args_are_set: bool = attrs.field(init=False, default=False)
    args: Iterable = attrs.field(init=False, factory=list)
    kwargs: Dict = attrs.field(init=False, factory=dict)

    completion_callbacks: list[Callable] = attrs.field(factory=list)
    upkeep_settings: TaskUpkeepSettings = attrs.field(factory=TaskUpkeepSettings)
    execution_id: Optional[str] = attrs.field(init=False, default=None)

    status: TaskStatus = TaskStatus.NOT_SUBMITTED
    outcome: Optional[TaskOutcome[R_co]] = None

    curr_retry: Optional[int] = None
    max_retry: int = 3
    # cache_expiration: datetime.timedelta = None

    def add_tags(self, tags: list[str]) -> Task:  # pragma: no cover
        self.tags += tags
        return self

    # Split into __init__ and _set_up because ParamSpec doesn't allow us
    # to play with kwargs.
    # cc: https://peps.python.org/pep-0612/#concatenating-keyword-parameters
    def _set_up(self, *args: Iterable, **kwargs: Dict):
        assert not self.args_are_set
        self.args = args
        self.kwargs = kwargs
        self.args_are_set = True

    def __call__(self) -> TaskOutcome[R_co]:
        self.status = TaskStatus.RUNNING
        if self.upkeep_settings.perform_upkeep:
            outcome = self._call_with_upkeep()
        else:
            outcome = self._call_without_upkeep()

        self.outcome = outcome
        logger.debug(f"DONE: Execution of {self}.")

        if self.outcome.exception is None:
            self.status = TaskStatus.SUCCEEDED
        else:
            self.status = TaskStatus.FAILED

        if self.status == TaskStatus.SUCCEEDED or (
            self.curr_retry is not None and self.curr_retry >= self.max_retry
        ):
            logger.debug("Running completion callbacks...")
            for callback in self.completion_callbacks:
                callback(task=self)
        else:
            logger.debug(
                f"Not running completion callbacks: retry {self.curr_retry}/{self.max_retry}."
            )

        return outcome

    def cancel_without_starting(self):
        logger.debug(f"Cancelling task {self} without starting")
        self.status = TaskStatus.FAILED
        self.outcome = TaskOutcome(exception=exceptions.MazepaCancel())
        for callback in self.completion_callbacks:
            callback(task=self)

    def _call_with_upkeep(self) -> TaskOutcome[R_co]:
        assert self.upkeep_settings.interval_secs is not None

        def _perform_upkeep_callbacks():
            for fn in self.upkeep_settings.callbacks:
                fn()

        upkeep = RepeatTimer(self.upkeep_settings.interval_secs, _perform_upkeep_callbacks)
        upkeep.start()
        result = self._call_without_upkeep()
        upkeep.cancel()
        return result

    def _call_without_upkeep(self) -> TaskOutcome[R_co]:
        assert self.args_are_set
        logger.debug(f"STARTING: Execution of {self}.")
        time_start = time.time()
        try:
            # TODO: parametrize by task execution environment
            return_value = self.fn(*self.args, **self.kwargs)
            exception = None
            traceback_text = None
            logger.debug("Successful task execution.")
        except (exceptions.MazepaException, SystemExit, KeyboardInterrupt) as exc:
            raise exc  # pragma: no cover
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"Failed task execution of {self}.")
            logger.exception(exc)
            exc_type, exception, tb = sys.exc_info()
            traceback_text = "".join(traceback.format_exception(exc_type, exception, tb))
            return_value = None

        time_end = time.time()

        outcome = TaskOutcome(
            exception=exception,
            traceback_text=traceback_text,
            execution_secs=time_end - time_start,
            return_value=return_value,
        )

        return outcome


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

    def get_operation_name(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> str:
        ...

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
    operation_name: str = "Unclassified Task"
    id_fn: Callable[[Callable, list, dict], str] = attrs.field(
        default=functools.partial(id_generation.generate_invocation_id, prefix="task")
    )
    tags: list[str] = attrs.field(factory=list)
    time_bound: bool = True
    max_retry: int = 3

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
        upkeep_settings = TaskUpkeepSettings(
            perform_upkeep=(not self.time_bound),
            interval_secs=5,
        )
        result = Task[R_co](
            fn=self.fn,
            operation_name=self.operation_name,
            id_=id_,
            tags=self.tags,
            upkeep_settings=upkeep_settings,
            max_retry=self.max_retry,
        )
        result._set_up(*args, **kwargs)  # pylint: disable=protected-access # friend class
        return result


@overload
def taskable_operation(
    fn: Callable[P, R_co], *, time_bound: bool = ...
) -> TaskableOperation[P, R_co]:
    ...


@overload
def taskable_operation(
    *, time_bound: bool = ...
) -> Callable[[Callable[P, R_co]], TaskableOperation[P, R_co]]:
    ...


def taskable_operation(fn=None, *, operation_name=None, time_bound: bool = True):
    if fn is not None:
        if operation_name is None:
            operation_name = fn.__name__
        return _TaskableOperation(fn=fn, operation_name=operation_name, time_bound=time_bound)
    else:
        return cast(
            Callable[[Callable[P, R_co]], TaskableOperation[P, R_co]],
            functools.partial(
                taskable_operation,
                operation_name=operation_name,
                time_bound=time_bound,
            ),
        )


def taskable_operation_cls(cls: Type[RawTaskableOperationCls]):
    def _make_task(self, *args, **kwargs):
        task = _TaskableOperation(  # pylint: disable=protected-access
            self,
        ).make_task(
            *args, **kwargs
        )
        task.operation_name = self.get_operation_name(
            *args, **kwargs
        )
        return task


    # can't override __new__ because it doesn't combine well with attrs/dataclass
    setattr(cls, "make_task", _make_task)
    return cls

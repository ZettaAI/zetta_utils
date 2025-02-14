from __future__ import annotations

import functools
import sys
import time
import traceback
import uuid
from concurrent.futures import TimeoutError as PebbleTimeoutError
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Protocol,
    Type,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)

import attrs
from pebble import concurrent
from typing_extensions import ParamSpec

from zetta_utils import log

from . import constants, exceptions, id_generation
from .task_outcome import TaskOutcome, TaskStatus

logger = log.get_logger("mazepa")

R_co = TypeVar("R_co", covariant=True)
P = ParamSpec("P")


@attrs.mutable
class TaskUpkeepSettings:
    perform_upkeep: bool = False
    interval_sec: float | None = None


@attrs.mutable
class Task(Generic[R_co]):  # pylint: disable=too-many-instance-attributes
    """
    An executable task.
    """

    fn: Callable[..., R_co]
    operation_name: str = "Unnamed Task"
    id_: str = attrs.field(factory=lambda: str(uuid.uuid1()))
    worker_type: str | None = None

    args: Iterable = attrs.field(factory=list)
    kwargs: dict = attrs.field(factory=dict)

    upkeep_settings: TaskUpkeepSettings = attrs.field(factory=TaskUpkeepSettings)
    execution_id: str | None = attrs.field(init=False, default=None)

    status: TaskStatus = TaskStatus.NOT_SUBMITTED
    outcome: TaskOutcome[R_co] | None = None
    runtime_limit_sec: float | None = None

    # cache_expiration: datetime.timedelta = None
    # to play with kwargs.
    # cc: https://peps.python.org/pep-0612/#concatenating-keyword-parameters
    def _set_up(self, *args: Iterable, **kwargs: dict):
        self.args = args
        self.kwargs = kwargs

    def _call_task_fn(self, debug: bool = True) -> R_co:
        if debug or self.runtime_limit_sec is None:
            return_value = self.fn(*self.args, **self.kwargs)
        else:
            future = concurrent.process(timeout=self.runtime_limit_sec)(self.fn)(
                *self.args, **self.kwargs
            )
            try:
                return_value = future.result()
            except PebbleTimeoutError as e:
                raise exceptions.MazepaTimeoutError(f"Task '{self.id_}' took too long.") from e
        return return_value

    def __call__(self, debug: bool = True, handle_exceptions: bool = True) -> TaskOutcome[R_co]:
        logger.debug(f"STARTING: Execution of {self}.")
        self.status = TaskStatus.RUNNING
        time_start = time.time()

        exception = None
        traceback_text = None
        if handle_exceptions:
            try:
                return_value = self._call_task_fn(debug=debug)
                logger.debug("Successful task execution.")
            except (SystemExit, KeyboardInterrupt) as exc:  # pragma: no cover
                raise exc
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(f"Failed task execution of {self}.")
                logger.exception(exc)
                exc_type, exception, tb = sys.exc_info()
                traceback_text = "".join(traceback.format_exception(exc_type, exception, tb))
                return_value = None
        else:
            return_value = self._call_task_fn(debug=debug)

        time_end = time.time()
        logger.info(f"Task done in: {time_end - time_start:.2f}sec.")

        outcome = TaskOutcome(
            exception=exception,
            traceback_text=traceback_text,
            execution_sec=time_end - time_start,
            return_value=return_value,
        )

        self.outcome = outcome
        logger.debug(f"DONE: Execution of {self}.")

        if self.outcome.exception is None:
            self.status = TaskStatus.SUCCEEDED
        else:
            self.status = TaskStatus.FAILED

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
    worker_type: str | None = None
    runtime_limit_sec: float | None = None
    upkeep_interval_sec: float = constants.DEFAULT_UPKEEP_INTERVAL

    def __call__(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R_co:
        return self.fn(*args, **kwargs)

    def make_task(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Task[R_co]:
        id_ = self.id_fn(self.fn, list(args), kwargs)
        upkeep_settings = TaskUpkeepSettings(
            perform_upkeep=True,  # All tasks perform upkeep now
            interval_sec=self.upkeep_interval_sec,
        )
        result = Task[R_co](
            fn=self.fn,
            operation_name=self.operation_name,
            id_=id_,
            worker_type=self.worker_type,
            upkeep_settings=upkeep_settings,
            runtime_limit_sec=self.runtime_limit_sec,
        )
        result._set_up(*args, **kwargs)  # pylint: disable=protected-access # friend class
        return result


@overload
def taskable_operation(
    fn: Callable[P, R_co],
    *,
    runtime_limit_sec: float | None = ...,
    operation_name: str | None = ...,
) -> TaskableOperation[P, R_co]:
    ...


@overload
def taskable_operation(
    *,
    runtime_limit_sec: float | None = ...,
    operation_name: str | None = ...,
) -> Callable[[Callable[P, R_co]], TaskableOperation[P, R_co]]:
    ...


def taskable_operation(
    fn=None,
    *,
    runtime_limit_sec: float | None = None,
    operation_name: str | None = None,
):
    if fn is not None:
        if operation_name is None:
            operation_name = fn.__name__
        return _TaskableOperation(
            fn=fn,
            runtime_limit_sec=runtime_limit_sec,
            operation_name=operation_name,
        )
    else:
        return cast(
            Callable[[Callable[P, R_co]], TaskableOperation[P, R_co]],
            functools.partial(
                _TaskableOperation,
                runtime_limit_sec=runtime_limit_sec,
                operation_name=operation_name,
            ),
        )


def taskable_operation_cls(
    cls: Type[RawTaskableOperationCls] | None = None,
    *,
    operation_name: str | None = None,
):
    def _make_task(self, *args, **kwargs):
        if operation_name is None:
            if hasattr(self, "get_operation_name"):
                operation_name_final = self.get_operation_name()  # pragma: no cover
            else:
                operation_name_final = type(self).__name__
        else:
            operation_name_final = operation_name
        task = _TaskableOperation(  # pylint: disable=protected-access
            self,
            operation_name=operation_name_final,
            # TODO: Other params passed to decorator
        ).make_task(
            *args, **kwargs
        )  # pylint: disable=protected-access
        return task

    if cls is not None:
        # can't override __new__ because it doesn't combine well with attrs/dataclass
        setattr(cls, "make_task", _make_task)
        return cls
    else:
        return cast(
            Callable[
                [Type[RawTaskableOperationCls]],
                Any,  # Any bc mypy ignores class decorator return type
            ],
            functools.partial(
                taskable_operation_cls,
                operation_name=operation_name,
            ),
        )

from __future__ import annotations

import functools
import sys
import threading
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
import tenacity
from pebble import concurrent
from typing_extensions import ParamSpec

from zetta_utils import log

from . import constants, exceptions, id_generation
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
    interval_sec: float | None = None
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
    kwargs: dict = attrs.field(init=False, factory=dict)

    completion_callbacks: list[Callable] = attrs.field(factory=list)
    upkeep_settings: TaskUpkeepSettings = attrs.field(factory=TaskUpkeepSettings)
    execution_id: str | None = attrs.field(init=False, default=None)

    status: TaskStatus = TaskStatus.NOT_SUBMITTED
    outcome: TaskOutcome[R_co] | None = None
    runtime_limit_sec: float | None = None

    # cache_expiration: datetime.timedelta = None

    def add_tags(self, tags: list[str]) -> Task:  # pragma: no cover
        self.tags += tags
        return self

    # Split into __init__ and _set_up because ParamSpec doesn't allow us
    # to play with kwargs.
    # cc: https://peps.python.org/pep-0612/#concatenating-keyword-parameters
    def _set_up(self, *args: Iterable, **kwargs: dict):
        assert not self.args_are_set
        self.args = args
        self.kwargs = kwargs
        self.args_are_set = True

    def __call__(self, debug: bool = True) -> TaskOutcome[R_co]:
        self.status = TaskStatus.RUNNING
        if self.upkeep_settings.perform_upkeep:
            outcome = self._call_with_upkeep(debug=debug)
        else:
            # All tasks perform upkeep for now, so no need to test this case
            outcome = self._call_without_upkeep(debug=debug)  # pragma: no cover

        self.outcome = outcome
        logger.debug(f"DONE: Execution of {self}.")

        if self.outcome.exception is None:
            run_completion_callbacks = True
            self.status = TaskStatus.SUCCEEDED
        else:
            run_completion_callbacks = True
            self.status = TaskStatus.FAILED
            if isinstance(self.outcome.exception, PebbleTimeoutError):
                logger.debug(f"Task {self.id_} execution timed out!")
                # Abstract the pebble library from the user
                self.outcome.exception = TimeoutError(str(self.outcome.exception))
                run_completion_callbacks = False

        if run_completion_callbacks:
            logger.debug("Running completion callbacks...")
            for callback in self.completion_callbacks:
                callback(task=self)

        return outcome

    def cancel_without_starting(self):
        logger.debug(f"Cancelling task {self} without starting")
        self.status = TaskStatus.FAILED
        self.outcome = TaskOutcome(exception=exceptions.MazepaCancel())
        for callback in self.completion_callbacks:
            callback(task=self)

    def _call_with_upkeep(self, debug: bool) -> TaskOutcome[R_co]:
        assert self.upkeep_settings.interval_sec is not None

        def _perform_upkeep_callbacks():
            try:
                for fn in self.upkeep_settings.callbacks:
                    fn()
            except tenacity.RetryError as e:
                logger.info(f"Couldn't perform upkeep: {e}")

        upkeep = RepeatTimer(self.upkeep_settings.interval_sec, _perform_upkeep_callbacks)
        upkeep.start()
        try:
            result = self._call_without_upkeep(debug=debug)
        except Exception as e:  # pragma: no cover
            raise e from None
        finally:
            upkeep.cancel()

        return result

    def _call_without_upkeep(self, debug: bool) -> TaskOutcome[R_co]:
        assert self.args_are_set
        logger.debug(f"STARTING: Execution of {self}.")
        time_start = time.time()

        try:
            if debug or self.runtime_limit_sec is None:
                return_value = self.fn(*self.args, **self.kwargs)
            else:
                future = concurrent.process(timeout=self.runtime_limit_sec)(self.fn)(
                    *self.args, **self.kwargs
                )
                return_value = future.result()

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
            execution_sec=time_end - time_start,
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
    runtime_limit_sec: float | None = None
    upkeep_interval_sec: float = constants.DEFAULT_UPKEEP_INTERVAL

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
            perform_upkeep=True,  # All tasks perform upkeep now
            interval_sec=self.upkeep_interval_sec,
        )
        result = Task[R_co](
            fn=self.fn,
            operation_name=self.operation_name,
            id_=id_,
            tags=self.tags,
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
    *, runtime_limit_sec: float | None = ..., operation_name: str | None = ...
) -> Callable[[Callable[P, R_co]], TaskableOperation[P, R_co]]:
    ...


def taskable_operation(
    fn=None, *, runtime_limit_sec: float | None = None, operation_name: str | None = None
):
    if fn is not None:
        if operation_name is None:
            operation_name = fn.__name__
        return _TaskableOperation(
            fn=fn, runtime_limit_sec=runtime_limit_sec, operation_name=operation_name
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
    cls: Type[RawTaskableOperationCls] | None = None, *, operation_name: str | None = None
):
    def _make_task(self, *args, **kwargs):
        if operation_name is None:
            if hasattr(self, "get_operation_name"):
                operation_name_final = self.get_operation_name()  # pragma: no cover # viz
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

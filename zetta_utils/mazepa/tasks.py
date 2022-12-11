from __future__ import annotations

import functools
import time
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
from pathos.multiprocessing import ProcessingPool
from typing_extensions import ParamSpec

from zetta_utils import log

from . import id_generation
from .exceptions import MazepaStopException
from .task_outcome import TaskOutcome, TaskStatus

logger = log.get_logger("mazepa")

R_co = TypeVar("R_co", covariant=True)
P = ParamSpec("P")


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

    def _call_with_upkeep(self) -> TaskOutcome[R_co]:
        assert self.upkeep_settings.interval_secs is not None

        pool = ProcessingPool(processes=1)
        async_result = pool.apipe(self._call_without_upkeep)

        while True:
            if async_result.ready():
                break

            for callback in self.upkeep_settings.callbacks:
                callback()

            time.sleep(self.upkeep_settings.interval_secs)

        return_val = async_result.get()
        return return_val

    def _call_without_upkeep(self) -> TaskOutcome[R_co]:
        assert self.args_are_set
        logger.debug(f"STARTING: Execution of {self}.")
        time_start = time.time()
        try:
            # TODO: parametrize by task execution environment
            return_value = self.fn(*self.args, **self.kwargs)
            exception = None
            logger.debug("Successful task execution.")
        except (MazepaStopException, SystemExit, KeyboardInterrupt) as exc:
            raise exc  # pragma: no cover
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"Failed task execution of {self}.")
            logger.exception(exc)
            exception = exc
            return_value = None

        time_end = time.time()
        logger.debug(f"DONE: Execution of {self}.")

        outcome = TaskOutcome(
            exception=exception,
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
            interval_secs=10,
        )
        result = Task[R_co](
            fn=self.fn,
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


def taskable_operation(fn=None, *, time_bound: bool = True):
    if fn is not None:
        return _TaskableOperation(fn=fn)
    else:
        return cast(
            Callable[[Callable[P, R_co]], TaskableOperation[P, R_co]],
            functools.partial(_TaskableOperation, time_bound=time_bound),
        )


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

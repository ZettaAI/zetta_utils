# pylint: disable=global-statement,redefined-outer-name,unused-argument
from __future__ import annotations

import functools
from contextlib import AbstractContextManager
from typing import Any, Iterable
from unittest.mock import MagicMock

import pytest

from zetta_utils.mazepa import (
    Dependency,
    InMemoryExecutionState,
    TaskStatus,
    concurrent_flow,
    execute,
    flow_schema,
    taskable_operation,
)
from zetta_utils.mazepa.autoexecute_task_queue import AutoexecuteTaskQueue
from zetta_utils.mazepa.exceptions import MazepaExecutionFailure, MazepaTimeoutError
from zetta_utils.mazepa.execution import Executor
from zetta_utils.mazepa.task_outcome import OutcomeReport
from zetta_utils.mazepa.tasks import Task
from zetta_utils.mazepa.transient_errors import (
    MAX_TRANSIENT_RETRIES,
    ExplicitTransientError,
)
from zetta_utils.message_queues.base import MessageQueue, ReceivedMessage

TASK_COUNT = 0


class DummyWrapperQueue(MessageQueue):
    name: str = "wrapper for testing parallel acknowledgement"
    queue: AutoexecuteTaskQueue = AutoexecuteTaskQueue(debug=True)

    def push(self, payloads: Iterable[Task]):
        self.queue.push(payloads)

    def pull(self, max_num: int = 1) -> list[ReceivedMessage[OutcomeReport]]:
        return self.queue.pull(max_num)


@pytest.fixture
def reset_task_count():
    global TASK_COUNT
    TASK_COUNT = 0


@taskable_operation
def dummy_task(argument: str) -> Any:
    global TASK_COUNT
    TASK_COUNT += 1
    return f"return-for-{argument}"


@flow_schema
def dummy_flow(argument: str):
    task1 = dummy_task.make_task(argument=f"{argument}-x1")
    yield task1
    yield Dependency(task1)
    assert task1.status == TaskStatus.SUCCEEDED
    assert task1.outcome is not None
    assert task1.outcome.return_value == f"return-for-{argument}-x1"
    task2 = dummy_task.make_task(argument=f"{argument}-x2")
    yield task2


@flow_schema
def dummy_flow_without_outcome(argument: str):
    task1 = dummy_task.make_task(argument=f"{argument}-x1")
    yield task1
    yield Dependency(task1)
    task2 = dummy_task.make_task(argument=f"{argument}-x2")
    yield task2


@flow_schema
def dummy_flow2():
    task1 = dummy_task.make_task(argument="x1")
    yield task1
    yield Dependency(task1)
    task2 = dummy_task.make_task(argument="x2")
    yield task2


@flow_schema
def empty_flow():
    yield []


def test_local_execution_defaults(reset_task_count):
    execute(
        concurrent_flow(
            [
                dummy_flow("f1"),
                dummy_flow("f2"),
                dummy_flow("f3"),
            ]
        ),
        batch_gap_sleep_sec=0,
        max_batch_len=2,
        do_dryrun_estimation=False,
    )
    assert TASK_COUNT == 6


def test_local_execution_one_flow(reset_task_count):
    execute(
        dummy_flow("f1"),
        batch_gap_sleep_sec=0,
        max_batch_len=1,
        do_dryrun_estimation=False,
    )
    assert TASK_COUNT == 2


def test_local_executor_one_flow(reset_task_count):
    Executor(
        batch_gap_sleep_sec=0,
        max_batch_len=1,
        do_dryrun_estimation=False,
    )(dummy_flow("f1"))
    assert TASK_COUNT == 2


def test_local_execution_one_callable(reset_task_count) -> None:
    execute(
        functools.partial(dummy_task, argument="x0"),
        batch_gap_sleep_sec=0,
        max_batch_len=1,
        do_dryrun_estimation=False,
    )
    assert TASK_COUNT == 1


def test_local_execution_one_task(reset_task_count) -> None:
    execute(
        dummy_task.make_task(argument="x0"),
        batch_gap_sleep_sec=0,
        do_dryrun_estimation=False,
        max_batch_len=1,
    )
    assert TASK_COUNT == 1


def test_local_execution_with_dryrun(reset_task_count):
    execute(dummy_flow2(), max_batch_len=1, do_dryrun_estimation=True, show_progress=False)
    assert TASK_COUNT == 2


def make_mock_ctx_mngr(mocker) -> AbstractContextManager[Any]:
    mngr_m = mocker.NonCallableMock(spec=AbstractContextManager)
    mngr_m.__enter__ = mocker.MagicMock()
    mngr_m.__exit__ = mocker.MagicMock()
    return mngr_m


def test_local_execution_state(reset_task_count):
    execute(
        InMemoryExecutionState(
            [
                dummy_flow("f1"),
                dummy_flow("f2"),
                dummy_flow("f3"),
            ]
        ),
        execution_id="yo",
        batch_gap_sleep_sec=0,
        max_batch_len=2,
        do_dryrun_estimation=False,
    )
    assert TASK_COUNT == 6


def test_local_execution_state_queue(reset_task_count):
    q = AutoexecuteTaskQueue(debug=True)
    execute(
        InMemoryExecutionState(
            [
                dummy_flow("f1"),
                dummy_flow("f2"),
                dummy_flow("f3"),
            ]
        ),
        task_queue=q,
        outcome_queue=q,
        batch_gap_sleep_sec=0,
        do_dryrun_estimation=False,
        max_batch_len=2,
    )
    assert TASK_COUNT == 6


def test_local_no_sleep(mocker):
    sleep_m = mocker.patch("time.sleep")
    execute(
        empty_flow(),
        batch_gap_sleep_sec=10,
        max_batch_len=2,
        do_dryrun_estimation=False,
    )
    sleep_m.assert_not_called()


def test_non_local_sleep(mocker):
    sleep_m = mocker.patch("time.sleep")
    queue_m = mocker.MagicMock(spec=MessageQueue)
    execute(
        empty_flow(),
        batch_gap_sleep_sec=10,
        max_batch_len=2,
        do_dryrun_estimation=False,
        task_queue=queue_m,
        outcome_queue=queue_m,
    )
    sleep_m.assert_called_once()


def test_local_execution_backup_write(reset_task_count, mocker):
    record_execution_checkpoint_m = mocker.patch(
        "zetta_utils.mazepa.execution.record_execution_checkpoint"
    )

    execute(
        concurrent_flow(
            [
                dummy_flow("f1"),
                dummy_flow("f2"),
                dummy_flow("f3"),
            ]
        ),
        batch_gap_sleep_sec=0,
        max_batch_len=2,
        do_dryrun_estimation=False,
        checkpoint_interval_sec=0.0,
    )
    record_execution_checkpoint_m.assert_called()


def test_local_execution_backup_read(reset_task_count, mocker):
    task1 = dummy_task.make_task(argument="f1-x1")
    mocker.patch(
        "zetta_utils.mazepa.execution_state.read_execution_checkpoint",
        return_value=set([task1.id_]),
    )

    execute(
        concurrent_flow(
            [
                dummy_flow_without_outcome("f1"),
                dummy_flow_without_outcome("f2"),
                dummy_flow_without_outcome("f3"),
            ]
        ),
        batch_gap_sleep_sec=0,
        max_batch_len=2,
        do_dryrun_estimation=False,
        checkpoint="MOCKED_PATH",
    )


def test_autoexecute_task_error(mocker):
    q = AutoexecuteTaskQueue(handle_exceptions=True, debug=False)
    task_fn: MagicMock = mocker.MagicMock(side_effect=[Exception, 10])
    task = Task(task_fn)
    with pytest.raises(MazepaExecutionFailure):
        execute(
            target=task,
            task_queue=q,
            outcome_queue=q,
            batch_gap_sleep_sec=0,
            do_dryrun_estimation=False,
            max_batch_len=2,
        )


def test_autoexecute_task_transient_error(mocker):
    q = AutoexecuteTaskQueue(handle_exceptions=True)
    task_fn: MagicMock = mocker.MagicMock(side_effect=[ExplicitTransientError(), 10])
    task = Task(task_fn)
    execute(
        target=task,
        task_queue=q,
        outcome_queue=q,
        batch_gap_sleep_sec=0,
        do_dryrun_estimation=False,
        max_batch_len=2,
    )
    assert task.status == TaskStatus.SUCCEEDED
    assert task.outcome is not None and task.outcome.return_value == 10
    assert task_fn.call_count == 2


def test_autoexecute_task_timeout_retry(mocker):
    q = AutoexecuteTaskQueue(handle_exceptions=True)
    task = Task(mocker.MagicMock(side_effect=[MazepaTimeoutError, 10]))

    execute(
        target=task,
        task_queue=q,
        outcome_queue=q,
        batch_gap_sleep_sec=0,
        do_dryrun_estimation=False,
        max_batch_len=2,
    )
    assert task.status == TaskStatus.SUCCEEDED
    assert task.outcome is not None and task.outcome.return_value == 10


def test_autoexecute_task_transient_error_too_many(mocker):
    q = AutoexecuteTaskQueue(handle_exceptions=True)
    task_fn: MagicMock = mocker.MagicMock(
        side_effect=[ExplicitTransientError] * (MAX_TRANSIENT_RETRIES + 1) + [10]
    )
    task = Task(task_fn)
    with pytest.raises(MazepaExecutionFailure):
        execute(
            target=task,
            task_queue=q,
            outcome_queue=q,
            batch_gap_sleep_sec=0,
            do_dryrun_estimation=False,
            max_batch_len=2,
        )


def test_parallel_acknowledgement(reset_task_count):
    q = DummyWrapperQueue()
    execute(
        dummy_flow("f1"),
        task_queue=q,
        outcome_queue=q,
        batch_gap_sleep_sec=0,
        max_batch_len=1,
        do_dryrun_estimation=False,
    )
    assert TASK_COUNT == 2

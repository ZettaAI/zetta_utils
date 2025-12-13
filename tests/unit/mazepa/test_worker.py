import time
from unittest.mock import MagicMock

import pytest

from zetta_utils.mazepa.pool_activity import PoolActivityTracker
from zetta_utils.mazepa.tasks import _TaskableOperation
from zetta_utils.mazepa.worker import run_worker
from zetta_utils.message_queues.base import ReceivedMessage


class MockQueue:
    def __init__(self):
        self.messages = []
        self.pushed_messages = []

    def pull(self, max_num=1):
        if len(self.messages) == 0:
            return []
        result = self.messages[:max_num]
        self.messages = self.messages[max_num:]
        return result

    def push(self, messages):
        self.pushed_messages.extend(messages)


@pytest.fixture
def mock_queues():
    task_queue = MockQueue()
    outcome_queue = MockQueue()
    return task_queue, outcome_queue


@pytest.fixture
def pool_activity_tracker():
    pool_name = "test_worker_pool"
    tracker = PoolActivityTracker(pool_name)
    tracker.create_shared_memory().close()
    yield tracker, pool_name
    tracker.unlink()


def test_worker_without_activity_tracker(mock_queues):  # pylint: disable=redefined-outer-name
    task_queue, outcome_queue = mock_queues

    def simple_task():
        return "done"

    task = _TaskableOperation(simple_task).make_task()
    msg = ReceivedMessage(
        payload=task,
        acknowledge_fn=MagicMock(),
        extend_lease_fn=MagicMock(),
        approx_receive_count=1,
    )
    task_queue.messages = [msg]

    result = run_worker(
        task_queue=task_queue,
        outcome_queue=outcome_queue,
        sleep_sec=0.1,
        max_runtime=0.5,
        debug=True,
    )

    assert result == "max_runtime_exceeded"
    assert len(outcome_queue.pushed_messages) == 1
    msg.acknowledge_fn.assert_called_once()  # type: ignore[attr-defined]


def test_worker_with_activity_tracker_updates_on_task_processing(
    mock_queues, pool_activity_tracker  # pylint: disable=redefined-outer-name
):
    task_queue, outcome_queue = mock_queues
    tracker, pool_name = pool_activity_tracker

    def simple_task():
        return "done"

    task = _TaskableOperation(simple_task).make_task()
    msg = ReceivedMessage(
        payload=task,
        acknowledge_fn=MagicMock(),
        extend_lease_fn=MagicMock(),
        approx_receive_count=1,
    )
    task_queue.messages = [msg]

    last_activity_before, active_count_before = tracker.get_activity_data()
    assert active_count_before == 0

    time.sleep(0.1)

    result = run_worker(
        task_queue=task_queue,
        outcome_queue=outcome_queue,
        sleep_sec=0.1,
        max_runtime=0.5,
        debug=True,
        pool_name=pool_name,
    )

    last_activity_after, active_count_after = tracker.get_activity_data()

    assert result == "max_runtime_exceeded"
    assert len(outcome_queue.pushed_messages) == 1
    msg.acknowledge_fn.assert_called_once()  # type: ignore[attr-defined]
    assert last_activity_after > last_activity_before
    assert active_count_after == 0


def test_worker_with_activity_tracker_idle_timeout(
    mock_queues, pool_activity_tracker  # pylint: disable=redefined-outer-name
):
    task_queue, outcome_queue = mock_queues
    _, pool_name = pool_activity_tracker

    result = run_worker(
        task_queue=task_queue,
        outcome_queue=outcome_queue,
        sleep_sec=0.1,
        max_runtime=5.0,
        debug=True,
        idle_timeout=0.3,
        pool_name=pool_name,
    )

    assert result == "idle_timeout_exceeded"
    assert len(outcome_queue.pushed_messages) == 0


def test_worker_with_activity_tracker_no_idle_when_tasks_processing(
    mock_queues, pool_activity_tracker  # pylint: disable=redefined-outer-name
):
    task_queue, outcome_queue = mock_queues
    _, pool_name = pool_activity_tracker

    def slow_task():
        time.sleep(0.15)
        return "done"

    task = _TaskableOperation(slow_task).make_task()
    msg = ReceivedMessage(
        payload=task,
        acknowledge_fn=MagicMock(),
        extend_lease_fn=MagicMock(),
        approx_receive_count=1,
    )
    task_queue.messages = [msg]

    result = run_worker(
        task_queue=task_queue,
        outcome_queue=outcome_queue,
        sleep_sec=0.05,
        max_runtime=0.4,
        debug=True,
        idle_timeout=0.3,
        pool_name=pool_name,
    )

    assert result == "max_runtime_exceeded"
    assert len(outcome_queue.pushed_messages) == 1
    msg.acknowledge_fn.assert_called_once()  # type: ignore[attr-defined]


def test_worker_activity_tracker_active_count_increments_decrements(
    mock_queues, pool_activity_tracker  # pylint: disable=redefined-outer-name
):
    task_queue, outcome_queue = mock_queues
    tracker, pool_name = pool_activity_tracker

    active_counts = []

    def task_that_checks_active_count():
        _, active_count = tracker.get_activity_data()
        active_counts.append(active_count)
        return "done"

    task = _TaskableOperation(task_that_checks_active_count).make_task()
    msg = ReceivedMessage(
        payload=task,
        acknowledge_fn=MagicMock(),
        extend_lease_fn=MagicMock(),
        approx_receive_count=1,
    )
    task_queue.messages = [msg]

    _, active_count_before = tracker.get_activity_data()
    assert active_count_before == 0

    result = run_worker(
        task_queue=task_queue,
        outcome_queue=outcome_queue,
        sleep_sec=0.1,
        max_runtime=0.5,
        debug=True,
        pool_name=pool_name,
    )

    _, active_count_after = tracker.get_activity_data()

    assert result == "max_runtime_exceeded"
    assert len(outcome_queue.pushed_messages) == 1
    assert active_counts[0] == 1
    assert active_count_after == 0


def test_worker_without_pool_name_no_activity_tracking(
    mock_queues,  # pylint: disable=redefined-outer-name
):
    task_queue, outcome_queue = mock_queues

    def simple_task():
        return "done"

    task = _TaskableOperation(simple_task).make_task()
    msg = ReceivedMessage(
        payload=task,
        acknowledge_fn=MagicMock(),
        extend_lease_fn=MagicMock(),
        approx_receive_count=1,
    )
    task_queue.messages = [msg]

    result = run_worker(
        task_queue=task_queue,
        outcome_queue=outcome_queue,
        sleep_sec=0.1,
        max_runtime=0.5,
        debug=True,
        idle_timeout=0.2,
        pool_name=None,
    )

    assert result == "max_runtime_exceeded"
    assert len(outcome_queue.pushed_messages) == 1

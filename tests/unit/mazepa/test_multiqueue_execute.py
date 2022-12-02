from __future__ import annotations

import time

import pytest

from zetta_utils.mazepa import ExecutionMultiQueue
from zetta_utils.mazepa.tasks import Task

from .maker_utils import make_test_task


def test_constructor(mocker):
    queue_a = mocker.MagicMock()
    queue_b = mocker.MagicMock()
    queue_a.name = "a"
    queue_b.name = "b"
    meq = ExecutionMultiQueue([queue_a, queue_b])
    assert queue_a.name in meq.name
    assert queue_b.name in meq.name


def test_push_tasks(mocker):
    queue_a = mocker.MagicMock()
    queue_b = mocker.MagicMock()
    queue_a.name = "a"
    queue_b.name = "b"
    meq = ExecutionMultiQueue([queue_a, queue_b])
    task_a = make_test_task(lambda: None, id_="dummy", tags=["a"])
    task_b = make_test_task(lambda: None, "dummy", tags=["b"])
    task_bb = make_test_task(lambda: None, "dummy", tags=["b"])
    meq.push_tasks([task_a, task_b, task_bb])
    queue_a.push_tasks.assert_called_with([task_a])
    queue_b.push_tasks.assert_called_with([task_b, task_bb])


def test_push_tasks_exc(mocker):
    queue_a = mocker.MagicMock()
    queue_b = mocker.MagicMock()
    queue_a.name = "a"
    queue_b.name = "b"
    meq = ExecutionMultiQueue([queue_a, queue_b])
    task_c = Task(lambda: None, "dummy", tags=["c"])
    with pytest.raises(RuntimeError):
        meq.push_tasks([task_c])


def test_pull_task_outcomes_no_max(mocker):
    queue_a = mocker.MagicMock()
    queue_b = mocker.MagicMock()
    queue_a.name = "a"
    queue_b.name = "b"

    outcomes_a = {f"a_{i}": mocker.MagicMock() for i in range(3)}
    queue_a.pull_task_outcomes = mocker.MagicMock(return_value=outcomes_a)
    outcomes_b = {f"b_{i}": mocker.MagicMock() for i in range(3)}
    queue_b.pull_task_outcomes = mocker.MagicMock(return_value=outcomes_b)
    meq = ExecutionMultiQueue([queue_a, queue_b])
    result = meq.pull_task_outcomes(max_num=4000)
    for k in {**outcomes_a, **outcomes_b}.keys():
        assert k in result


def test_pull_task_outcomes_max_num(mocker):
    queue_a = mocker.MagicMock()
    queue_b = mocker.MagicMock()
    queue_a.name = "a"
    queue_b.name = "b"

    outcomes_a = {f"a_{i}": mocker.MagicMock() for i in range(3)}
    queue_a.pull_task_outcomes = mocker.MagicMock(return_value=outcomes_a)
    outcomes_b = {f"b_{i}": mocker.MagicMock() for i in range(3)}
    queue_b.pull_task_outcomes = mocker.MagicMock(return_value=outcomes_b)
    meq = ExecutionMultiQueue([queue_a, queue_b])
    result = meq.pull_task_outcomes(max_num=1)

    for k in outcomes_a.keys():
        assert k in result
    for k in outcomes_b.keys():
        assert k not in result


def test_pull_task_outcomes_max_time(mocker):
    queue_a = mocker.MagicMock()
    queue_b = mocker.MagicMock()
    queue_a.name = "a"
    queue_b.name = "b"

    outcomes_a = {f"a_{i}": mocker.MagicMock() for i in range(3)}

    def slow_return(*kargs, **kwargs):  # pylint: disable=unused-argument
        time.sleep(0.01)
        return outcomes_a

    queue_a.pull_task_outcomes = mocker.MagicMock(side_effect=slow_return)
    outcomes_b = {f"b_{i}": mocker.MagicMock() for i in range(3)}
    queue_b.pull_task_outcomes = mocker.MagicMock(return_value=outcomes_b)
    meq = ExecutionMultiQueue([queue_a, queue_b])
    result = meq.pull_task_outcomes(max_num=4000, max_time_sec=0.005)

    for k in outcomes_a.keys():
        assert k in result
    for k in outcomes_b.keys():
        assert k not in result


def test_pull_tasks(mocker):
    queues = []
    tasks = {}
    for name in "a", "b", "c":
        queue = mocker.MagicMock()
        queue.name = name
        q_tasks = [mocker.MagicMock(), mocker.MagicMock()]
        queue.pull_tasks = mocker.MagicMock(return_value=q_tasks)
        tasks[name] = q_tasks
        queues.append(queue)

    meq = ExecutionMultiQueue(queues)
    result = meq.pull_tasks(max_num=4)

    for k in "a", "b":
        for e in tasks[k]:
            assert e in result

    for k in "c":
        for e in tasks[k]:
            assert e not in result

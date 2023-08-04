from __future__ import annotations

import pytest

from zetta_utils.mazepa import TaskRouter
from zetta_utils.mazepa.tasks import Task

from .maker_utils import make_test_task


def test_constructor(mocker):
    queue_a = mocker.MagicMock()
    queue_b = mocker.MagicMock()
    queue_a.name = "a"
    queue_b.name = "b"
    meq = TaskRouter([queue_a, queue_b])
    assert queue_a.name in meq.name
    assert queue_b.name in meq.name


def test_push_tasks(mocker):
    queue_a = mocker.MagicMock()
    queue_b = mocker.MagicMock()
    queue_a.name = "a"
    queue_b.name = "b"
    meq = TaskRouter([queue_a, queue_b])
    task_a = make_test_task(lambda: None, id_="dummy", tags=["a"])
    task_b = make_test_task(lambda: None, "dummy", tags=["b"])
    task_bb = make_test_task(lambda: None, "dummy", tags=["b"])
    meq.push([task_a, task_b, task_bb])
    queue_a.push.assert_called_with([task_a])
    queue_b.push.assert_called_with([task_b, task_bb])


def test_push_tasks_exc(mocker):
    queue_a = mocker.MagicMock()
    queue_b = mocker.MagicMock()
    queue_a.name = "a"
    queue_b.name = "b"
    meq = TaskRouter([queue_a, queue_b])
    task_c = Task(lambda: None, "dummy", tags=["c"])
    with pytest.raises(RuntimeError):
        meq.push([task_c])

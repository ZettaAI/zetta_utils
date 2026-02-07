from __future__ import annotations

import pytest

from zetta_utils.mazepa import TaskRouter
from zetta_utils.mazepa.tasks import Task

from .maker_utils import make_test_task


def test_constructor(mocker):
    queue_a = mocker.MagicMock()
    queue_b = mocker.MagicMock()
    queue_a.name = "run-xxx_type-a_work"
    queue_b.name = "run-xxx_type-b_work"
    meq = TaskRouter([queue_a, queue_b])
    assert queue_a.name in meq.name
    assert queue_b.name in meq.name


def test_push_tasks(mocker):
    queue_a = mocker.MagicMock()
    queue_b = mocker.MagicMock()
    queue_a.name = "run-xxx_type-a_work"
    queue_b.name = "run-xxx_type-b_work"
    meq = TaskRouter([queue_a, queue_b])
    task_a = make_test_task(lambda: None, id_="dummy").with_worker_type("type_a")
    task_b = make_test_task(lambda: None, "dummy").with_worker_type("type_b")
    task_bb = make_test_task(lambda: None, "dummy").with_worker_type("type_b")
    meq.push([task_a, task_b, task_bb])
    queue_a.push.assert_called_with([task_a])
    queue_b.push.assert_called_with([task_b, task_bb])


def test_push_tasks_exc(mocker):
    queue_a = mocker.MagicMock()
    queue_b = mocker.MagicMock()
    queue_a.name = "run-xxx_type-a_work"
    queue_b.name = "run-xxx_type-b_work"
    meq = TaskRouter([queue_a, queue_b])
    task_c = Task(lambda: None, "dummy", worker_type="type_c")
    with pytest.raises(RuntimeError):
        meq.push([task_c])


def test_push_tasks_mem_vs_mem_agg(mocker):
    """Test that mem and mem_agg worker types route to correct queues."""
    queue_mem = mocker.MagicMock()
    queue_mem_agg = mocker.MagicMock()
    queue_mem.name = "run-xxx_mem_work"
    queue_mem_agg.name = "run-xxx_mem-agg_work"
    meq = TaskRouter([queue_mem, queue_mem_agg])

    task_mem = make_test_task(lambda: None, id_="task1").with_worker_type("mem")
    task_mem_agg = make_test_task(lambda: None, id_="task2").with_worker_type("mem_agg")

    meq.push([task_mem, task_mem_agg])

    queue_mem.push.assert_called_with([task_mem])
    queue_mem_agg.push.assert_called_with([task_mem_agg])

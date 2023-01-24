from __future__ import annotations

import time

import attrs

from zetta_utils.mazepa import (
    Task,
    TaskableOperation,
    taskable_operation,
    taskable_operation_cls,
)


def test_make_taskable_operation_cls() -> None:
    @taskable_operation_cls
    @attrs.mutable
    class DummyTaskCls:
        def __call__(self) -> str:
            return "result"

    obj: TaskableOperation[[], str] = DummyTaskCls()
    obj = DummyTaskCls()

    assert isinstance(obj, TaskableOperation)
    task = obj.make_task()
    assert isinstance(task, Task)
    outcome = obj()
    assert outcome == "result"


def test_make_taskable_operation() -> None:
    @taskable_operation
    def dummy_task_fn():
        pass

    assert isinstance(dummy_task_fn, TaskableOperation)
    task = dummy_task_fn.make_task()
    assert isinstance(task, Task)


def test_task_runtime_limit() -> None:
    @taskable_operation(runtime_limit_sec=0.1)
    def dummy_task_fn():
        time.sleep(0.3)

    assert isinstance(dummy_task_fn, TaskableOperation)
    task = dummy_task_fn.make_task()
    assert isinstance(task, Task)
    outcome = task(debug=False)
    assert isinstance(outcome.exception, TimeoutError)

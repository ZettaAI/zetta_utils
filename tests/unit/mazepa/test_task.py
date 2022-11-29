from __future__ import annotations

import attrs

from zetta_utils.mazepa import Task, TaskableOperation, taskable_operation, taskable_operation_cls


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

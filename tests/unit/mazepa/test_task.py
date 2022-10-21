from __future__ import annotations
import attrs
from zetta_utils.mazepa import task_factory, Task, task_factory_cls, TaskFactory


def test_make_task_factory_cls() -> None:
    @task_factory_cls
    @attrs.mutable
    class DummyTaskCls:
        def __call__(self) -> str:
            return "result"

    obj: TaskFactory[[], str] = DummyTaskCls()
    obj = DummyTaskCls()

    assert isinstance(obj, TaskFactory)
    task = obj.make_task()
    assert isinstance(task, Task)
    outcome = obj()
    assert outcome == "result"


def test_make_task_factory() -> None:
    @task_factory
    def dummy_task_fn():
        pass

    assert isinstance(dummy_task_fn, TaskFactory)
    task = dummy_task_fn.make_task()
    assert isinstance(task, Task)

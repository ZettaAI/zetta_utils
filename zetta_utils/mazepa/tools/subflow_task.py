from __future__ import annotations
from typing import Generic
from typing_extensions import ParamSpec
from .. import task_factory_cls
from .. import Flow, Executor

P = ParamSpec("P")


@task_factory_cls
class SubflowTask(Generic[P]):
    subflow: Flow[P]
    executor: Executor

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:  # pragma: no cover
        self.executor(self.subflow)

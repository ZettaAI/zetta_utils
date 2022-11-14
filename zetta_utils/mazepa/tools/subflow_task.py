from __future__ import annotations

from typing_extensions import ParamSpec

from .. import Executor, Flow, task_factory_cls

P = ParamSpec("P")


@task_factory_cls
class SubflowTask:
    subflow: Flow
    executor: Executor

    def __call__(self) -> None:  # pragma: no cover
        self.executor(self.subflow)

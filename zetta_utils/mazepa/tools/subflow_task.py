from __future__ import annotations

from typing_extensions import ParamSpec

from .. import Executor, Flow, taskable_operation

P = ParamSpec("P")


@taskable_operation
class SubflowTask:
    subflow: Flow
    executor: Executor

    def __call__(self) -> None:  # pragma: no cover
        self.executor(self.subflow)

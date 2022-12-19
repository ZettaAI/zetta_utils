from __future__ import annotations

from typing import Generic

from typing_extensions import ParamSpec

from .. import Executor, FlowSchema, taskable_operation

P = ParamSpec("P")


@taskable_operation
class SubflowTask(Generic[P]):
    subflow: FlowSchema[P]
    executor: Executor

    def __call__(self) -> None:  # pragma: no cover
        self.executor(self.subflow)

from __future__ import annotations

import attrs


@attrs.mutable
class TaskExecutionEnv:
    tags: list[str] = attrs.field(factory=list)

    def extend(self, other: TaskExecutionEnv):
        raise NotImplementedError  # pragma: no cover

    def apply_defaults(self, other: TaskExecutionEnv):
        raise NotImplementedError  # pragma: no cover

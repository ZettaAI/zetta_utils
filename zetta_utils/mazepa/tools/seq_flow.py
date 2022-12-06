from __future__ import annotations

from typing import Union

from .. import Dependency, Flow, Task, flow_schema


@flow_schema
def seq_flow(stages: list[Union[Flow, Task]]):  # pragma: no cover
    for e in stages:
        yield e
        yield Dependency()

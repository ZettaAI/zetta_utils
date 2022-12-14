from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExecutionCtxManager(Protocol):
    def __call__(self, execution_id: str) -> AbstractContextManager[Any]:
        ...

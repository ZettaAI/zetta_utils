from enum import Enum, auto, unique
from typing import Generic, Optional, TypeVar

import attrs


@unique
class TaskStatus(Enum):
    NOT_SUBMITTED = auto()
    SUBMITTED = auto()
    RUNNING = auto()
    SUCCEEDED = auto()
    FAILED = auto()


R_co = TypeVar("R_co", covariant=True)


@attrs.mutable
class TaskOutcome(Generic[R_co]):
    exception: Optional[BaseException] = None
    traceback_text: Optional[str] = None
    execution_secs: Optional[float] = None
    return_value: Optional[R_co] = None

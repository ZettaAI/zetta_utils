"""Task management subpackage — lazily resolved via zetta_utils.common.lazy."""
from typing import TYPE_CHECKING

from zetta_utils.common.lazy import make_lazy_module

_LAZY_SUBPACKAGES = (
    "types",
    "user",
    "task",
    "dependency",
    "timesheet",
    "metrics",
    "task_type",
    "project",
    "exceptions",
    "task_types",
)

__getattr__, __dir__ = make_lazy_module(__name__, globals(), _LAZY_SUBPACKAGES)

if TYPE_CHECKING:
    from . import (
        dependency,
        exceptions,
        metrics,
        project,
        task,
        task_type,
        task_types,
        timesheet,
        types,
        user,
    )

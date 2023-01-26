from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

from zetta_utils import log

task_id = ContextVar[Optional[str]]("task_id", default=None)
execution_id = ContextVar[Optional[str]]("execution_id", default=None)

log.CTX_VARS["task_id"] = task_id
log.CTX_VARS["execution_id"] = execution_id

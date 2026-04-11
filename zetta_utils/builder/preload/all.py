# pylint: disable=unused-import, wrong-import-position
"""All module imports."""

import time

from zetta_utils import log

_start = time.perf_counter()

from zetta_utils.builder.preload import inference
from zetta_utils.builder.preload import training
from zetta_utils import task_management

_elapsed = time.perf_counter() - _start
log.get_logger("zetta_utils").debug(f"Preload all modules: {_elapsed:.2f}s")

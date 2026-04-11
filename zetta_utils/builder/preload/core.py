# pylint: disable=unused-import, wrong-import-position
"""Core module imports - shared by all load modes."""

import time

_start = time.perf_counter()

from zetta_utils import log, typing, parsing, builder, common, constants
from zetta_utils import geometry, distributions, layer, ng

# Add builder module suppression now that it's loaded
log.add_supress_traceback_module(builder)

_elapsed = time.perf_counter() - _start
log.get_logger("zetta_utils").debug(f"Preload core modules: {_elapsed:.2f}s")

# pylint: disable=unused-import, broad-exception-caught
"""Try-load variant with error handling."""

import builtins
import ctypes

from zetta_utils import log

logger = log.get_logger("zetta_utils")


def _install_cuda_driver_state_tracer():
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        logger.warning(f"cuda driver tracer: could not load libcuda.so.1: {e}")
        return

    def _get_state():
        count = ctypes.c_int(0)
        return libcuda.cuDeviceGetCount(ctypes.byref(count))

    last_state = _get_state()
    logger.info(f"cuda driver tracer: initial cuDeviceGetCount={last_state}")
    original_import = builtins.__import__

    def traced_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        result = original_import(name, globals_, locals_, fromlist, level)
        nonlocal last_state
        new_state = _get_state()
        if new_state != last_state:
            from_str = f" from={list(fromlist)}" if fromlist else ""
            logger.info(
                f"cuda driver tracer: state CHANGED during import {name!r}"
                f"{from_str}: {last_state} -> {new_state}"
            )
            last_state = new_state
        return result

    builtins.__import__ = traced_import


_install_cuda_driver_state_tracer()


try:
    from zetta_utils.builder.preload import core
except Exception as e:
    logger.exception(e)

try:
    from zetta_utils.builder.preload import inference
except Exception as e:
    logger.exception(e)

try:
    from zetta_utils.builder.preload import training
except Exception as e:
    logger.exception(e)

try:
    from zetta_utils import mazepa_addons
except Exception as e:
    logger.exception(e)

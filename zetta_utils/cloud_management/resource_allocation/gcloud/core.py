from typing import Any

from google.api_core.extended_operation import ExtendedOperation

from zetta_utils import log

logger = log.get_logger("zetta_utils")


def wait_for_extended_operation(op: ExtendedOperation, timeout: int = 300) -> Any:
    """
    Waits for the extended (long-running) operation to complete.
    """
    result = op.result(timeout=timeout)
    if op.error_code:
        logger.warning(f"Error during {op.name}: [Code: {op.error_code}]: {op.error_message}")
        raise op.exception() or RuntimeError(op.error_message)

    if op.warnings:
        for warning in op.warnings:
            logger.warning(f"{warning.code}: {warning.message}")
    return result

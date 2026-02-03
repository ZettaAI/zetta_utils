# pylint: disable=unused-import, broad-exception-caught
"""Try-load variant with error handling."""

from zetta_utils import log

logger = log.get_logger("zetta_utils")

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

"""Entry point for ``python -m zetta_utils.run.gc``."""

import logging

from zetta_utils.run.gc import logger, main

if __name__ == "__main__":  # pragma: no cover
    logger.setLevel(logging.INFO)
    main()

"""Entry point for ``python -m zetta_utils.run.gc``."""

import logging

from zetta_utils.log import get_logger
from zetta_utils.run.gc.orchestrator import main

if __name__ == "__main__":  # pragma: no cover
    get_logger("zetta_utils").setLevel(logging.INFO)
    main()

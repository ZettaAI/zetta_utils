"""Output logging."""
# pylint: disable=global-statement
from __future__ import annotations

import logging

from rich.logging import RichHandler


def get_time_str(log_time):
    return log_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


SAVED_LEVEL = "ERROR"

def configure_logger(level=None, third_party_level="WARN"):
    for _ in (
        "python_jsonschema_objects",
        "pytorch_lightning",
        "urllib3",
        "google",
        "gcsfs",
        "fsspec",
        "asyncio",
        "botocore",
        "matplotlib",
        "git",
        "h5py",
        "torch",
    ):
        logging.getLogger(_).setLevel(third_party_level)

    global SAVED_LEVEL
    if level is None:
        level = SAVED_LEVEL
    else:
        if isinstance(level, int):
            level = max(0, 20 - 10 * level)
        SAVED_LEVEL = level

    logging.basicConfig(
        level=level,
        format="'%(name)s' %(pathname)20s:%(lineno)4d \n%(message)s",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                show_path=False,
                show_time=True,
                enable_link_path=False,
                log_time_format=get_time_str,
            )
        ],
    )
    logging.getLogger("mazepa").setLevel(level)
    logging.getLogger("zetta_utils").setLevel(level)


def get_logger(name):
    configure_logger()
    return logging.getLogger(name)

def set_verbosity(verbosity_level):
    global SAVED_LEVEL
    SAVED_LEVEL = verbosity_level
    logging.getLogger("zetta_utils").setLevel(verbosity_level)
    logging.getLogger("mazepa").setLevel(verbosity_level)

configure_logger()
#logger = logging.getLogger("zetta_utils")

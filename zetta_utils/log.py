"""Output logging."""
# pylint: disable=global-statement
from __future__ import annotations

import logging
import os
from contextvars import ContextVar
from typing import Optional

import cachetools
import logging_loki
import typeguard
from rich.logging import RichHandler
from rich.traceback import install

SUPRESS_TRACEBACK_MODULES = [cachetools, typeguard]

CTX_VARS = {k: ContextVar[Optional[str]](k, default=None) for k in ["zetta_user", "zetta_project"]}

GRAPHANA_KEY = os.environ.get("GRAPHANA_CLOUD_ACCESS_KEY", None)
if GRAPHANA_KEY is not None:
    LOKI_HANDLER = logging_loki.LokiHandler(
        url=f"https://334581:{GRAPHANA_KEY}@logs-prod3.grafana.net/loki/api/v1/push", version="1"
    )
else:
    LOKI_HANDLER = None


def set_logging_label(name, value):
    CTX_VARS[name].set(value)


class InjectingFilter(logging.Filter):
    """
    A filter which injects context-specific information into logs
    """

    def __init__(self):
        super().__init__()

    def filter(self, record):
        for k, v in CTX_VARS.items():
            value = v.get()
            if LOKI_HANDLER is not None:
                LOKI_HANDLER.emitter.tags[k] = value
            setattr(record, k, value)
        return True


def update_traceback():
    install(show_locals=True, suppress=SUPRESS_TRACEBACK_MODULES)


def add_supress_traceback_module(module):
    SUPRESS_TRACEBACK_MODULES.append(module)
    update_traceback()


update_traceback()


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

    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        show_time=True,
        enable_link_path=False,
        log_time_format=get_time_str,
    )
    rich_handler.addFilter(InjectingFilter())
    handlers = [rich_handler]
    if LOKI_HANDLER is not None:
        handlers.append(LOKI_HANDLER)
    logging.basicConfig(
        level=level, format="%(name)s %(pathname)20s:%(lineno)4d \n%(message)s", handlers=handlers
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
# logger = logging.getLogger("zetta_utils")

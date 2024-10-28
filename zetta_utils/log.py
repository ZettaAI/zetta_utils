"""Output logging."""
# pylint: disable=global-statement
from __future__ import annotations

import logging
import os
import pickle
from contextlib import contextmanager
from typing import Any

import attr
import attrs
import cachetools
import dill
import logging_loki
import typeguard
from rich.logging import RichHandler
from rich.traceback import install

SUPRESS_TRACEBACK_MODULES = [cachetools, typeguard, attr, attrs, pickle, dill]

LOKI_HANDLER: logging_loki.LokiHandler | None = None


class InjectingFilter(logging.Filter):
    """
    A filter which injects context-specific information into logs
    """

    def __init__(self):
        super().__init__()

    def filter(self, record):
        for k in ["zetta_user", "zetta_project"]:
            value = CTX_VARS[k]
            if LOKI_HANDLER is not None:
                LOKI_HANDLER.emitter.tags[k] = value
            setattr(record, k, value)
        return True


def update_traceback():
    install(show_locals=False, suppress=SUPRESS_TRACEBACK_MODULES)


def add_supress_traceback_module(module):
    SUPRESS_TRACEBACK_MODULES.append(module)
    update_traceback()


update_traceback()


def get_time_str(log_time):
    return log_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


SAVED_LEVEL = "ERROR"


def configure_logger(level=None, third_party_level="ERROR"):
    """
    Configures the logging system with customizable settings.

    :param level: The logging level to set for the root logger. If None, uses the
        previously saved logging level. If an integer is provided, it will be
        interpreted as a custom logging level. If a string is provided, it should
        be one of the logging level names (e.g., 'DEBUG', 'INFO', 'WARNING',
        'ERROR', 'CRITICAL'). Defaults to None; saved level defaults to 'ERROR'
        unless changed via the :func:`set_verbosity` function.
    :param third_party_level: The logging level to set for third-party libraries.
        Defaults to 'ERROR', meaning only error-level logs from third-party
        libraries will be displayed.

    .. note::
        - The logging levels for specific third-party libraries can be further
        configured within this function.
        - The logging configuration includes settings for rich traceback formatting,
        injecting context-specific information into logs, and selecting appropriate
        logging handlers (stream handler for local execution or rich handler for
        Kubernetes execution).

    :returns: None
    """

    for _ in (
        "python_jsonschema_objects",
        "pytorch_lightning",
        "lightning_fabric",
        "urllib3",
        "urllib3.connectionpool",
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

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)s %(pathname)s:%(lineno)d: %(message)s")
    stream_handler.setFormatter(formatter)

    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        show_time=True,
        enable_link_path=False,
        log_time_format=get_time_str,
        tracebacks_word_wrap=False,
    )
    rich_handler.addFilter(InjectingFilter())
    try:
        _ = os.environ["KUBERNETES_SERVICE_HOST"]
        handlers = [stream_handler]
    except KeyError:
        handlers = [rich_handler]  # type: ignore
    # TODO: Add Loki Handler with multiprocessing support / deferred write
    # if LOKI_HANDLER is not None:
    #    handlers.append(LOKI_HANDLER)
    logging.basicConfig(
        level=level, format="%(name)s %(pathname)20s:%(lineno)4d \n%(message)s", handlers=handlers
    )
    logging.getLogger("mazepa").setLevel(level)
    logging.getLogger("zetta_utils").setLevel(level)


def get_logger(name):
    """
    Get a logger with the specified name, creating it if necessary.

    :param name: An identifying name (channel) for the logger to get. Use is up to
        the caller, but typically you would use the name of your project or code
        module. Zetta utils code uses "zetta_utils".

    .. note::
        - Each uniquely-named logger can have its own configuration settings (such
        as log level).
        - Call this method rather than calling :func:`getLogger` directly to ensure
        that the logging system is properly initialized.

    :returns: Logger instance
    """

    configure_logger()
    return logging.getLogger(name)


def set_verbosity(verbosity_level):
    """
    Set the log level of the zetta_utils and mazepa loggers, as well as the default
    verbosity for any subsequently-created loggers.

    :param verbosity_level: The logging level to set. If an integer is provided,
        it will be interpreted as a custom logging level. If a string is provided,
        it should be one of the logging level names (e.g., 'DEBUG', 'INFO', 'WARNING',
        'ERROR', 'CRITICAL').

    :returns: None
    """
    global SAVED_LEVEL
    SAVED_LEVEL = verbosity_level
    logging.getLogger("zetta_utils").setLevel(verbosity_level)
    logging.getLogger("mazepa").setLevel(verbosity_level)


ENV_CTX_VARS = [
    "zetta_user",
    "zetta_project",
    "my_node_name",
    "my_pod_name",
    "my_pod_ip",
    "my_pod_service_account",
]
# TODO: contextvars can't be `dill`-ed
# from contextvars import ContextVar
# CTX_VARS = {k: ContextVar[Optional[str]](k, default=None) for k in ENV_CTX_VARS}
CTX_VARS: dict[str, Any] = {k: None for k in ENV_CTX_VARS}


def set_logging_tag(name, value):
    CTX_VARS[name] = value


@contextmanager
def logging_tag_ctx(key, value):
    if key in CTX_VARS:
        old_value = CTX_VARS[key]
    else:
        old_value = None

    set_logging_tag(key, value)
    yield
    set_logging_tag(key, old_value)


def _init_ctx_vars():
    for k in CTX_VARS:
        k_env = k.upper()
        if k_env in os.environ:
            set_logging_tag(k, os.environ[k_env])


_init_ctx_vars()

GRAFANA_USER_ID = "340203"
GRAFANA_KEY = os.environ.get("GRAFANA_CLOUD_ACCESS_KEY", None)
if GRAFANA_KEY is not None:
    LOKI_HANDLER = logging_loki.LokiHandler(
        url=f"https://{GRAFANA_USER_ID}:{GRAFANA_KEY}@logs-prod3.grafana.net/loki/api/v1/push",
        version="1",
    )
configure_logger()

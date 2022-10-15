"""Output logging."""
from __future__ import annotations

import logging
from rich.logging import RichHandler


def get_time_str(log_time):
    return log_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def configure_logger(third_party_level="INFO"):
    for _ in (
        "python_jsonschema_objects",
        "urllib3",
        "google",
        "gcsfs",
        "fsspec",
        "asyncio",
        "botocore",
    ):
        logging.getLogger(_).setLevel(third_party_level)

    logging.basicConfig(
        level="NOTSET",
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


logger = logging.getLogger("zetta_utils")


def get_logger(name):
    configure_logger()
    return logging.getLogger(name)

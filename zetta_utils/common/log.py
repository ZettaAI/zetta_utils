"""Output logging."""
from __future__ import annotations

import logging
import time
import os
import pathlib
from datetime import datetime


logger = logging.getLogger("zetta")


def configure_logger(
    name: str = "zetta",
    verbose: bool = True,
    log_folder: str = "/tmp/logs/zetta_logs",
):
    pathlib.Path(log_folder).mkdir(parents=True, exist_ok=True)

    # clear to avoid double add
    logger.handlers = []
    log_level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(log_level)

    channel = logging.StreamHandler()
    channel.setLevel(log_level)
    info_format = (
        "[%(asctime)s.%(msecs)03d"
        ", pid%(process)6s"
        ", %(filename)20s:%(lineno)4d]"
        " %(levelname)6s"
        " - %(message)s"
    )
    time_format = "%m-%d %H:%M:%S"
    formatter = logging.Formatter(info_format, time_format)
    channel.setFormatter(formatter)
    logger.addHandler(channel)

    # This ts is for a file name, so it uses a different time format
    ts = datetime.utcfromtimestamp(int(time.time())).strftime("%Y-%m-%d-%H:%M:%S")
    file_handler = logging.FileHandler(os.path.join(log_folder, f"{name}.log.{ts}.yaml"))

    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)


configure_logger()

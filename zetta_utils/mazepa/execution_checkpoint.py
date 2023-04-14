from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import aiohttp.client_exceptions
import fsspec
import gcsfs.retry
import google.auth.exceptions
import requests.exceptions

from zetta_utils import log

logger = log.get_logger("mazepa")


EXECUTION_CHECKPOINT_PATH = "gs://zetta_utils_runs"
CHECKPOINT_COMPRESSION = "zstd"


def read_execution_checkpoint(
    filepath: str, ignore_prefix: Optional[Sequence[str]] = None
) -> set[str]:  # pragma: no cover
    """
    Read completed IDs from checkpoint file.

    `ignore_prefix`: skip IDs with matching prefix, e.g. `['flow-']`, default `None`.
    """
    if ignore_prefix is None:
        ignore_prefix = []

    logger.info(f"Reading execution checkpoint from {filepath}")
    with fsspec.open(filepath, "r", compression=CHECKPOINT_COMPRESSION) as f:
        completed_ids = json.load(f)
    assert isinstance(completed_ids, list)
    return set(
        id_ for id_ in completed_ids if not any(id_.startswith(prefix) for prefix in ignore_prefix)
    )


def record_execution_checkpoint(
    execution_id: str, ckpt_name: str, completed_ids: list[str], raise_on_error: bool = False
):  # pragma: no cover
    """
    Save completed tasks to checkpoint file
    """
    zetta_user = os.environ["ZETTA_USER"]
    info_path = os.environ.get("EXECUTION_CHECKPOINT_PATH", EXECUTION_CHECKPOINT_PATH)
    ckpt_path = os.path.join(
        info_path, zetta_user, execution_id, f"{ckpt_name}.{CHECKPOINT_COMPRESSION}"
    )

    logger.info(f"Saving execution checkpoint to {ckpt_path}")
    try:
        with fsspec.open(ckpt_path, "w", compression=CHECKPOINT_COMPRESSION) as f:
            json.dump(completed_ids, f, indent=2)

    except (
        requests.exceptions.RequestException,
        google.auth.exceptions.GoogleAuthError,
        aiohttp.client_exceptions.ClientError,
        # gcsfs doesn't have any useful base exception, gotta catch 'em all:
        # https://github.com/fsspec/gcsfs/blob/dda390af941b57b6911261e5c76d01cc3ddccb10/gcsfs/retry.py#L68-L105
        gcsfs.retry.HttpError,
        gcsfs.retry.ChecksumError,
        FileNotFoundError,
        OSError,
        ValueError,
        RuntimeError,
    ):
        if raise_on_error is True:
            raise
        logger.exception(f"Exception while saving checkpoint '{ckpt_name}'")

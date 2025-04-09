# pylint: disable=unused-argument
from __future__ import annotations

import uuid
from typing import Callable, Optional

import dill
import xxhash

from zetta_utils import log

logger = log.get_logger("mazepa")


def generate_invocation_id(
    fn: Optional[Callable] = None,
    args: Optional[list] = None,
    kwargs: Optional[dict] = None,
    prefix: Optional[str] = None,
) -> str:
    """Generate a unique and deterministic ID for a function invocation.
    The ID is generated using xxhash and dill to hash the function and its arguments.

    :param fn: the function, or really any Callable, defaults to None
    :param args: the function arguments, or any list, defaults to None
    :param kwargs: the function kwargs, or any dict, defaults to None
    :param prefix: optional prefix str, separated by `-`, defaults to None
    :return: A unique, yet deterministic string that identifies (fn, args, kwargs) in
      the current Python environment.
    """
    x = xxhash.xxh128()
    try:
        x.update(
            dill.dumps(
                (fn, args, kwargs),
                protocol=dill.DEFAULT_PROTOCOL,
                byref=False,
                recurse=True,
                fmode=dill.FILE_FMODE,
            )
        )
    except dill.PicklingError as e:
        logger.warning(f"Failed to pickle {fn} with args {args} and kwargs {kwargs}: {e}")
        x.update(str(uuid.uuid4()))

    if prefix is not None:
        return f"{prefix}-{x.hexdigest()}"
    else:
        return x.hexdigest()


def get_literal_id_fn(  # pylint: disable=unused-argument
    id_: str,
) -> Callable[[Callable, list, dict], str]:  # pragma: no cover
    def get_literal_id(  # pylint: disable=unused-argument
        fn: Callable,
        args: list,
        kwargs: dict,
    ) -> str:
        return id_

    return get_literal_id

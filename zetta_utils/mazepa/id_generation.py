# pylint: disable=unused-argument
from __future__ import annotations

import uuid
from typing import Callable, Optional

import xxhash
from coolname import generate_slug

from zetta_utils import log

logger = log.get_logger("mazepa")


def get_unique_id(
    slug_len: int = 3,
    add_uuid: bool = True,
    prefix: Optional[str] = None,
    max_len: int = 320,
) -> str:  # pragma: no cover
    slug = generate_slug(slug_len)

    if prefix is not None:
        result = f"{prefix}-{slug}"
    else:
        result = f"{slug}"

    if add_uuid:
        unique_id = str(uuid.uuid1())
        result += unique_id
    result = result[:max_len]
    while not result[-1].isalpha():
        result = result[:-1]
    return result


def _get_code_hash(
    fn: Callable, _hash: Optional[xxhash.xxh128] = None, _visited: Optional[set[int]] = None
) -> xxhash.xxh128:
    if _hash is None:
        _hash = xxhash.xxh128()
    if _visited is None:
        _visited = set()

    # Check to prevent infinite recursion
    # This is a bit silly, as the entire custom code hashing endeavor is done to avoid
    # issues with Python's code hash in the first place...
    # However, PYTHONHASHSEED is not an issue for tracking methods within the same session.
    # Generating recursive loops with the same code hash requires some effort by the user
    if id(fn) in _visited:
        return _hash

    _visited.add(id(fn))

    for attribute_name in {x for x in dir(fn) if not x.startswith("__")}:
        attrib = getattr(fn.__class__, attribute_name, None)

        if attrib is not None and isinstance(attrib, property):
            _get_code_hash(attrib, _hash, _visited)  # type: ignore
            continue

        attrib = getattr(fn, attribute_name)

        if callable(attrib):
            _get_code_hash(attrib, _hash, _visited)
        else:
            _hash.update(f"{attribute_name}: {attrib}".encode())

    if hasattr(fn, "__self__") and fn.__self__ is not None:
        _get_code_hash(fn.__self__, _hash, _visited)

        try:
            _get_code_hash(fn.__self__.__call__.__func__, _hash, _visited)
        except AttributeError:
            pass

    try:
        _hash.update(fn.__qualname__)
    except AttributeError:
        pass

    try:
        _hash.update(fn.__code__.co_code)
    except AttributeError:
        pass

    return _hash


def generate_invocation_id(
    fn: Callable,
    args: list,
    kwargs: dict,
    prefix: Optional[str] = None,
):
    # Decided against using Python `hash` due to randomized PYTHONHASHSEED, and
    # https://github.com/python/cpython/issues/94155 - esp. wrt to `co_code` missing.
    # Note that this check skips most code attributes, e.g. co_flags for performance reasons.

    x = _get_code_hash(fn)

    x.update(args.__repr__().encode())
    x.update(kwargs.__repr__().encode())

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

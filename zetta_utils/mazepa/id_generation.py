# pylint: disable=unused-argument
from __future__ import annotations

import functools
import uuid
from typing import Callable, Optional

import xxhash
from coolname import generate_slug

import zetta_utils.mazepa.tasks
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
    fn: Callable, _hash: Optional[xxhash.xxh128] = None, _prefix=""
) -> xxhash.xxh128:
    if _hash is None:
        _hash = xxhash.xxh128()

    try:
        _hash.update(fn.__qualname__)

        try:
            # Mypy wants to see (BuiltinFunctionType, MethodType, MethodWrapperType),
            # but not all have __self__.__dict__ that is not a mappingproxy
            method_kwargs = fn.__self__.__dict__  # type: ignore
            if isinstance(method_kwargs, dict):
                _hash.update(method_kwargs.__repr__())
        except AttributeError:
            pass

        _hash.update(fn.__code__.co_code)

        return _hash
    except AttributeError:
        pass

    if isinstance(fn, functools.partial):
        _hash.update(fn.args.__repr__().encode())

        _hash.update(fn.keywords.__repr__().encode())

        _hash = _get_code_hash(fn.func, _hash=_hash, _prefix=_prefix + "  ")
        return _hash

    if isinstance(
        fn, (zetta_utils.mazepa.tasks.TaskableOperation, zetta_utils.builder.BuilderPartial)
    ):
        _hash.update(fn.__repr__())

        _hash = _get_code_hash(fn.__call__, _hash=_hash, _prefix=_prefix + "  ")
        return _hash

    raise TypeError(f"Can't hash code for fn of type {type(fn)}")


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
    ) -> str:  # pragma: no cover
        return id_

    return get_literal_id

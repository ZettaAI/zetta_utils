# pylint: disable=unused-argument
from __future__ import annotations

import uuid
from typing import Callable, Optional

from coolname import generate_slug


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


def generate_invocation_id(
    fn: Callable,
    args: list,
    kwargs: dict,
    prefix: Optional[str] = None,
):  # pragma: no cover
    # TODO
    return get_unique_id(prefix=prefix)


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

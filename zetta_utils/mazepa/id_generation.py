# pylint: disable=unused-argument
import uuid
from typing import Callable, Optional

from coolname import generate_slug


def get_unique_id(
    slug_len: int = 3,
    prefix: Optional[str] = None,
) -> str:  # pragma: no cover
    slug = generate_slug(slug_len)
    unique_id = str(uuid.uuid1())

    if prefix is not None:
        result = f"{prefix}-{slug}-{unique_id}"
    else:
        result = f"{slug}-{unique_id}"

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

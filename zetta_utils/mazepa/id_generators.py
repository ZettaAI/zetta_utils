from typing import Callable
import uuid

from coolname import generate_slug  # type: ignore


def get_unique_id(  # pylint: disable=unused-argument
    fn: Callable,
    kwargs: dict,
    slug_len=3,
) -> str:  # pragma: no cover
    return f"{generate_slug(slug_len)}-{str(uuid.uuid1())}"


def get_literal_id_fn(  # pylint: disable=unused-argument
    id_: str,
) -> Callable[[Callable, dict], str]:  # pragma: no cover
    def get_literal_id(  # pylint: disable=unused-argument
        fn: Callable,
        kwargs: dict,
    ) -> str:  # pragma: no cover
        return id_

    return get_literal_id

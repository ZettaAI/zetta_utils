import uuid
from typing import Optional

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

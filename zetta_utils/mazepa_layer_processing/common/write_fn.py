from typing import Any

from zetta_utils import builder


@builder.register("write_fn")
def write_fn(src: Any):
    return src

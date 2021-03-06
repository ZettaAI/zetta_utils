# pylint: disable=all
from typing import Callable
from functools import partial
from typeguard import typechecked


@typechecked
def func_to_proc(func: Callable) -> Callable:  # pragma: no cover
    """Converting a vanilla function to buildable processor."""

    def wrapped(**kwargs):
        return partial(func, **kwargs)

    return wrapped

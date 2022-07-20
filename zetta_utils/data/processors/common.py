# pylint: disable=all
from typing import Callable
from functools import partial


def func_to_proc(func: Callable) -> Callable:
    """Converting a vanilla function to buildable processor."""

    def wrapped(**kwargs):
        return partial(func, **kwargs)

    return wrapped

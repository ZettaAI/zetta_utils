# pylint: disable=all
from functools import partial


def func_processor(func):
    def wrapped(**kwargs):
        return partial(func, **kwargs)

    return wrapped

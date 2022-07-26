# pylint: disable=all
import attrs
from typing import Callable
from typeguard import typechecked


@typechecked
@attrs.frozen
class ComparablePartial:
    func: Callable
    kwargs: dict = {}

    def __call__(self, **kwargs):
        return self.func(**self.kwargs, **kwargs)

def func_to_proc(func):
    def wrapped(**kwargs):
        return ComparablePartial(
            func=func,
            kwargs=kwargs,
        )

    return wrapped

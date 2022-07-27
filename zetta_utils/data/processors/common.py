# pylint: disable=all
import attrs
from typing import Callable, Any
from typeguard import typechecked


@typechecked
@attrs.frozen
class ComparablePartial:
    func: Callable
    kwargs: dict = {}

    def __call__(self, *args, **kwargs) -> Any:  # pragma: no cover
        return self.func(*args, **self.kwargs, **kwargs)


def func_to_proc(func):
    def wrapped(**kwargs):
        return ComparablePartial(
            func=func,
            kwargs=kwargs,
        )

    return wrapped

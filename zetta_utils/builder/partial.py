from typing import Callable, Any
from typeguard import typechecked
import attrs

# Need this class to make our processors comparable
@typechecked
@attrs.mutable(repr=False, init=False, slots=False)
class ComparablePartial:
    func: Callable
    kwargs: dict

    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs) -> Any:  # pragma: no cover
        return self.func(*args, **self.kwargs, **kwargs)

    def __repr__(self):  # pragma: no cover
        return f"ComparablePartial(func={self.func.__module__}.{self.func.__qualname__})"  # pragma: no cover # pylint: disable=line-too-long

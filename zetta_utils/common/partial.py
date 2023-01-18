from typing import Any, Callable, Dict, Generic, TypeVar

import attrs

R = TypeVar("R")

# Need this class to make our processors comparable
@attrs.mutable(repr=False, init=False, slots=False)
class ComparablePartial(Generic[R]):
    def __init__(self, func: Callable[..., R], **kwargs):
        self.func: Callable = func
        self.kwargs: Dict[str, Any] = kwargs

    @property
    def __name__(self) -> str:
        return self.func.__name__

    def __call__(self, *args, **kwargs) -> R:  # pragma: no cover
        return self.func(*args, **self.kwargs, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ComparablePartial(func={self.func.__module__}.{self.func.__qualname__}, kwargs={self.kwargs})"  # pragma: no cover # pylint: disable=line-too-long

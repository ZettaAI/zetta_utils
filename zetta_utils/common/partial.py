from typing import Any, Callable, Dict, Generic, TypeVar

import attrs

R = TypeVar("R")

# Need this class to make our processors comparable
@attrs.mutable(repr=False, init=False, slots=False)
class ComparablePartial(Generic[R]):
    def __init__(self, fn: Callable[..., R], **kwargs):
        self.fn: Callable = fn
        self.kwargs: Dict[str, Any] = kwargs

    @property
    def __name__(self) -> str:
        return self.fn.__name__

    def __call__(self, *args, **kwargs) -> R:  # pragma: no cover
        return self.fn(*args, **self.kwargs, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ComparablePartial(fn={self.fn.__module__}.{self.fn.__qualname__}, kwargs={self.kwargs})"  # pragma: no cover # pylint: disable=line-too-long

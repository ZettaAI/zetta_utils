from typing import Any, Callable, Dict, Generic, TypeVar

import attrs
from typeguard import typechecked

R = TypeVar("R")


@typechecked
@attrs.mutable(repr=False, init=False, slots=False)
class ComparablePartial(Generic[R]):
    """
    A pretty-printable, comparable implementation of a partial function specification.
    Only supports partially specifying keyword arguments.
    """

    def __init__(self, fn: Callable[..., R], **kwargs):
        self.fn: Callable = fn
        self.kwargs: Dict[str, Any] = kwargs

    def __call__(self, *args, **kwargs) -> R:  # pragma: no cover
        return self.fn(*args, **self.kwargs, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ComparablePartial(fn={self.fn.__module__}.{self.fn.__qualname__}, kwargs={self.kwargs})"  # pragma: no cover # pylint: disable=line-too-long

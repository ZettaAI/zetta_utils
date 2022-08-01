from typing import Callable, Any
import attrs
from typeguard import typechecked


# Need this class to make our processors comparable
@typechecked
@attrs.frozen
class ComparablePartial:
    func: Callable
    kwargs: dict = {}

    def __call__(self, *args, **kwargs) -> Any:  # pragma: no cover
        return self.func(*args, **self.kwargs, **kwargs)


# Need this class for a pretty print in the registry
@typechecked
@attrs.frozen(repr=False)
class FuncProcessorBuilder:
    func: Callable

    def __call__(self, **kwargs):
        return ComparablePartial(
            func=self.func,
            kwargs=kwargs,
        )

    def __repr__(self):  # pragma: no cover
        return f"FuncProcessorBuilder(func={self.func.__module__}.{self.func.__qualname__})"  # pragma: no cover # pylint: disable=line-too-long


def func_to_proc(func):
    builder = FuncProcessorBuilder(func)
    return builder

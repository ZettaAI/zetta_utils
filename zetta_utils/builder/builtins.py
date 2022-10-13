from typing import Callable
from . import register

@register("invoke")
def invoke(
    obj: Callable,
    params: dict,
    target: str = '__call__',
):
    return getattr(obj, target)(**params)

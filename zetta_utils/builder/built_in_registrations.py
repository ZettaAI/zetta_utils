from __future__ import annotations

from typing import Any, Callable

from .build import BuilderPartial
from .registry import register

LAMBDA_STR_MAX_LENGTH: int = 80


@register("lambda", False)
def efficient_parse_lambda_str(lambda_str: str) -> Callable:
    """Parses strings that are lambda functions"""
    if not isinstance(lambda_str, str):
        raise TypeError("`lambda_str` must be a string.")
    if not lambda_str.startswith("lambda"):
        raise ValueError("`lambda_str` must start with 'lambda'.")
    if len(lambda_str) > LAMBDA_STR_MAX_LENGTH:
        raise ValueError(f"`lambda_str` must be at most {LAMBDA_STR_MAX_LENGTH} characters.")

    return BuilderPartial(spec={"@type": "invoke_lambda_str", "lambda_str": lambda_str})


@register("invoke_lambda_str", False)
def invoke_lambda_str(*args: list, lambda_str: str, **kwargs: dict) -> Any:
    return eval(lambda_str)(*args, **kwargs)  # pylint: disable=eval-used


@register("chain")
def chain(data: Any, callables: list[Callable], chain_arg_name: str):
    result = data
    for e in callables:
        result = e(**{chain_arg_name: result})
    return result

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np  # pylint: disable=unused-import
import torch  # pylint: disable=unused-import


from .building import BuilderPartial
from .registry import register


@register("lambda", allow_partial=False)
def efficient_parse_lambda_str(lambda_str: str, name: Optional[str] = None) -> Callable:
    """Parses strings that are lambda functions"""
    if not isinstance(lambda_str, str):
        raise TypeError("`lambda_str` must be a string.")
    if not lambda_str.startswith("lambda"):
        raise ValueError("`lambda_str` must start with 'lambda'.")

    return BuilderPartial(spec={"@type": "invoke_lambda_str", "lambda_str": lambda_str}, name=name)


@register("invoke_lambda_str", allow_partial=False)
def invoke_lambda_str(*args: list, lambda_str: str, **kwargs: dict) -> Any:
    return eval(lambda_str)(*args, **kwargs)  # pylint: disable=eval-used

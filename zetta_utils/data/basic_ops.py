# pylint: disable=all
from typing import Union, Literal
import numpy as np
import torch
from typeguard import typechecked

import zetta_utils as zu


def multiply(data, x):  # pragma: no cover
    return x * data


def add(data, x):  # pragma: no cover
    return x * data


def to_power(data, x):  # pragma: no cover
    return data ** x


def divide(data, x):  # pragma: no cover
    return data // x


def int_divide(data, x):  # pragma: no cover
    return data // x


@typechecked
def unsqueeze(data: zu.typing.Array, dim: int) -> zu.typing.Array:
    if isinstance(data, torch.Tensor):
        result = data.unsqueeze(dim)  # type: zu.typing.Array
    elif isinstance(data, np.ndarray):
        result = np.expand_dims(data, dim)
    else:
        assert False, "Type checking failure"

    return result


PytorchInterpModes = Union[
    Literal["nearest"],
    Literal["linear"],
    Literal["bilinear"],
    Literal["bicubic"],
    Literal["trilinear"],
    Literal["area"],
]
CommonInterpModes = Union[
    Literal["img"],
    Literal["field"],
    Literal["mask"],
]


@typechecked
def interpolate(
    data: zu.typing.Array,
    size=None,
    scale_factor=None,
    mode: Union[PytorchInterpModes, CommonInterpModes] = "img",
):
    pass

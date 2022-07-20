# pylint: disable=all
from typing import Union, Literal, List, SupportsIndex, Tuple, Optional
import numpy as np
import torch
from typeguard import typechecked

import zetta_utils as zu


def multiply(data, x):  # pragma: no cover
    return x * data


def add(data, x):  # pragma: no cover
    return x * data


def power(data, x):  # pragma: no cover
    return data ** x


def divide(data, x):  # pragma: no cover
    return data / x


def int_divide(data, x):  # pragma: no cover
    return data // x


@typechecked
def unsqueeze(
    data: zu.typing.Array, dim: Union[SupportsIndex, Tuple[SupportsIndex, ...]]
) -> zu.typing.Array:
    if isinstance(data, torch.Tensor):
        if isinstance(dim, int):
            result = data.unsqueeze(dim)  # type: zu.typing.Array
        else:
            raise ValueError(
                f"Cannot use `torch.unsqueeze` with dim of type '{type(dim)}'"
            )
    elif isinstance(data, np.ndarray):
        result = np.expand_dims(data, dim)
    else:
        assert False, "Type checking failure"  # pragma: no cover

    return result


@typechecked
def squeeze(
    data: zu.typing.Array, dim: Union[SupportsIndex, Tuple[SupportsIndex, ...]] = None
) -> zu.typing.Array:
    if isinstance(data, torch.Tensor):
        if isinstance(dim, int) or dim is None:
            result = data.squeeze(dim)  # type: zu.typing.Array
        else:
            raise ValueError(
                f"Cannot use `torch.squeeze` with dim of type '{type(dim)}'"
            )
    elif isinstance(data, np.ndarray):
        if dim is not None:
            result = data.squeeze(dim)
        else:
            raise ValueError(f"Cannot use `np.squeeze` with dim of type '{type(dim)}'")
    else:
        assert False, "Type checking failure"  # pragma: no cover

    return result


TorchInterpolationMode = Literal[
    "nearest",
    "nearest-exact",
    "linear",
    "bilinear",
    "bicubic",
    "trilinear",
    "area",
]
CustomInterpolationMode = Literal[
    "img",
    "field",
    "mask",
    "segmentation",
]
InterpolationMode = Union[TorchInterpolationMode, CustomInterpolationMode]


@typechecked
def interpolate(
    data: zu.typing.Array,
    size=None,
    scale_factor: Union[float, List[float]] = None,
    mode: InterpolationMode = "img",
    mask_value_thr: float = 0,
):

    data_in = zu.data.convert.to_torch(data).float()
    if mode == "img" or mode == "field":
        interp_mode = "bilinear"
    elif mode == "mask":
        interp_mode = "area"
    elif mode == "segmentation":
        interp_mode = "nearest-exact"
        if (
            scale_factor is None
            or (isinstance(scale_factor, float) and scale_factor < 1.0)
            or (isinstance(scale_factor, list) and sum([i < 1.0 for i in scale_factor]))
            > 0
        ):
            raise NotImplementedError()
    else:
        interp_mode = mode

    raw_result = torch.nn.functional.interpolate(
        data_in,
        size=size,
        scale_factor=scale_factor,
        mode=interp_mode,
    )

    if mode == "field":
        assert scale_factor is not None
        assert isinstance(scale_factor, float)
        raw_result *= scale_factor
    elif mode == "mask":
        raw_result = raw_result > mask_value_thr
    elif mode == "segmentation":
        raw_result = raw_result.int()

    if isinstance(data, torch.Tensor):
        result = zu.data.convert.to_torch(raw_result)  # type: zu.typing.Array
    elif isinstance(data, np.ndarray):
        result = zu.data.convert.to_np(raw_result)
    else:
        assert False, "Type checker error."  # pragma: no cover

    return result


CompareMode = Literal[
    "eq",
    "==",
    "noeq",
    "!=",
    "gt",
    ">",
    "gte",
    ">=",
    "lt",
    "<",
    "lte",
    "<=",
]


@typechecked
def compare(
    data: zu.typing.Array,
    mode: CompareMode,
    operand: float,
    binarize: bool = True,
    fill: Optional[float] = None,
):
    if mode in ["eq", "=="]:
        mask = data == operand
    elif mode in ["neq", "!="]:
        mask = data != operand
    elif mode in ["gt", ">"]:
        mask = data > operand
    elif mode in ["gte", ">="]:
        mask = data >= operand
    elif mode in ["lt", "<"]:
        mask = data < operand
    elif mode in ["lte", "<="]:
        mask = data <= operand
    else:
        assert False, "Type checker failure."  # pragma: no cover

    if binarize:
        result = mask
    else:
        if fill is None:
            raise ValueError(
                "`fill` must be set to a floating point value when `binarize` == False"
            )
        result[mask] = fill

    return result

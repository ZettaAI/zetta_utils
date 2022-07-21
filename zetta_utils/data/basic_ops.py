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
    data: zu.typing.Array, dim: Union[SupportsIndex, Tuple[SupportsIndex, ...]] = 0
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
        result = data.squeeze(axis=dim)  # type: ignore # mypy doesn't see None is Ok
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


def _standardize_scale_factor(
    data_ndim: int,
    scale_factor: Optional[Union[float, List[float], Tuple[float, ...]]] = None,
    default_space_ndim: int = 2,
) -> Optional[Tuple[float, ...]]:
    if scale_factor is None:
        result = None
    else:
        data_space_ndim = data_ndim - 2  # Batch + Channel
        if isinstance(scale_factor, float):
            result = (scale_factor,) * min(data_space_ndim, default_space_ndim)
        else:
            result = tuple(scale_factor)

        while len(result) < data_space_ndim:
            result = (1.0,) + result

    return result


@typechecked
def interpolate(
    data: zu.typing.Array,
    size=None,
    scale_factor: Optional[Union[float, List[float], Tuple[float, ...]]] = None,
    mode: InterpolationMode = "img",
    mask_value_thr: float = 0,
    default_space_ndim: int = 2,
):
    scale_factor = _standardize_scale_factor(
        data_ndim=data.ndim,
        scale_factor=scale_factor,
        default_space_ndim=default_space_ndim,
    )

    if scale_factor is not None:
        spatial_ndim = len(scale_factor)
    else:
        if size is None:
            raise ValueError(
                "Neither `size` nor `scale_factor` provided to `interpolate()`"
            )
        spatial_ndim = len(size)

    if mode == "img" or mode == "field":
        if spatial_ndim == 3:
            interp_mode = "trilinear"
        elif spatial_ndim == 2:
            interp_mode = "bilinear"
        else:
            if spatial_ndim != 1:
                raise ValueError(
                    f"Unsupported number of spatial dimensions with data.shape = {data.shape}, "
                    f"scale_factor = {scale_factor}, size = {size}"
                )
            interp_mode = "linear"

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
            raise NotImplementedError()  # pragma: no cover
    else:
        interp_mode = mode

    data_in = zu.data.convert.to_torch(data).float()
    raw_result = torch.nn.functional.interpolate(
        data_in,
        size=size,
        scale_factor=scale_factor,
        mode=interp_mode,
    )

    if mode == "field":
        if scale_factor is None:
            raise NotImplementedError(  # pragma: no cover
                "`size`-based field interpolation is not currently supported."
            )

        if all(e == scale_factor[0] for e in scale_factor):
            multiplier = scale_factor[0]
        else:
            raise NotImplementedError(  # pragma: no cover
                "Non-isotropic field interpolation (scale_factor={scale_factor}) "
                "is not currently supported."
            )
            # For when we support non-isotropic scale factor
            # if raw_result.shape[1] != spatial_ndim:
            #     raise ValueError(
            #         f"Malformed field shape: {raw_result.shape}. Number "
            #         f"of channels ({raw_result.shape[1]}) must be equal to "
            #         f"the number of spatial dimensions ({spatial_ndim})."
            #     )

        raw_result *= multiplier
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
        if fill is not None:
            raise ValueError("`fill` must be set to None when `binarize` == True")

        result = mask
    else:
        if fill is None:
            raise ValueError(
                "`fill` must be set to a floating point value when `binarize` == False"
            )
        result = data
        result[mask] = fill

    return result

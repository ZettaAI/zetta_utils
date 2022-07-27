# pylint: disable=missing-docstring
import copy
from typing import Union, Literal, SupportsIndex, Optional, overload, Sequence
import numpy as np
import numpy.typing as npt
import torch
from typeguard import typechecked

import fastremap  # type: ignore
import cc3d  # type: ignore

import zetta_utils as zu
from zetta_utils.typing import Tensor


def multiply(data: Tensor, x) -> Tensor:  # pragma: no cover
    return x * data


def add(data: Tensor, x) -> Tensor:  # pragma: no cover
    return x * data


def power(data: Tensor, x) -> Tensor:  # pragma: no cover
    return data ** x


def divide(data: Tensor, x) -> Tensor:  # pragma: no cover
    return data / x


def int_divide(data: Tensor, x) -> Tensor:  # pragma: no cover
    return data // x


@overload
def unsqueeze(
    data: npt.NDArray, dim: Union[SupportsIndex, Sequence[SupportsIndex]] = ...
) -> npt.NDArray:  # pragma: no cover
    ...


@overload
def unsqueeze(
    data: torch.Tensor, dim: Union[SupportsIndex, Sequence[SupportsIndex]] = ...
) -> torch.Tensor:  # pragma: no cover
    ...


@typechecked
def unsqueeze(
    data: zu.typing.Tensor, dim: Union[SupportsIndex, Sequence[SupportsIndex]] = 0
) -> zu.typing.Tensor:
    if isinstance(data, torch.Tensor):
        if isinstance(dim, int):
            result = data.unsqueeze(dim)  # type: zu.typing.Tensor
        else:
            raise ValueError(f"Cannot use `torch.unsqueeze` with dim of type '{type(dim)}'")
    elif isinstance(data, np.ndarray):
        result = np.expand_dims(data, dim)
    else:
        assert False, "Type checking failure"  # pragma: no cover

    return result


@overload
def squeeze(
    data: npt.NDArray, dim: Optional[Union[SupportsIndex, Sequence[SupportsIndex]]] = ...
) -> npt.NDArray:  # pragma: no cover
    ...


@overload
def squeeze(
    data: torch.Tensor, dim: Optional[Union[SupportsIndex, Sequence[SupportsIndex]]] = ...
) -> torch.Tensor:  # pragma: no cover
    ...


@typechecked
def squeeze(
    data: zu.typing.Tensor, dim: Optional[Union[SupportsIndex, Sequence[SupportsIndex]]] = None
) -> zu.typing.Tensor:
    if isinstance(data, torch.Tensor):
        if isinstance(dim, int) or dim is None:
            result = data.squeeze(dim)  # type: zu.typing.Tensor
        else:
            raise ValueError(f"Cannot use `torch.squeeze` with dim of type '{type(dim)}'")
    else:
        assert isinstance(data, np.ndarray)
        result = data.squeeze(axis=dim)  # type: ignore # mypy thinkgs None is not ok, but it is

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
    scale_factor: Optional[Union[float, Sequence[float]]] = None,
    default_space_ndim: int = 2,
) -> Optional[Sequence[float]]:
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


def _get_spatial_ndim(
    size: Optional[Sequence[int]],
    scale_factor: Optional[Sequence[float]],
    data_shape: Sequence[int],
    allow_shape_rounding: bool,
) -> int:
    if scale_factor is not None:
        spatial_ndim = len(scale_factor)
        if not allow_shape_rounding:
            result_spatial_shape = [
                data_shape[2 + i] * scale_factor[i] for i in range(spatial_ndim)
            ]
            for i in range(spatial_ndim):
                if round(result_spatial_shape[i]) != result_spatial_shape[i]:
                    raise RuntimeError(
                        f"Interpolation of array with shape {data_shape} and scale "
                        "factor {scale_factor} would result in a non-integer shape "
                        f"along spatial dimention {i} "
                        f"({data_shape[2 + i]} -> {result_spatial_shape[i]}) while "
                        "`allow_shape_rounding` == False ."
                    )
    else:
        if size is None:
            raise ValueError("Neither `size` nor `scale_factor` provided to `interpolate()`")
        spatial_ndim = len(size)

    return spatial_ndim


def _get_torch_interp_mode(
    size: Optional[Sequence[int]],
    scale_factor: Optional[Sequence[float]],
    spatial_ndim: int,
    mode: InterpolationMode,
) -> TorchInterpolationMode:
    if mode in ("img", "field"):
        if spatial_ndim == 3:
            torch_interp_mode = "trilinear"  # type: TorchInterpolationMode
        elif spatial_ndim == 2:
            torch_interp_mode = "bilinear"
        else:
            if spatial_ndim != 1:
                raise RuntimeError(
                    f"Unsupported number of spatial dimensions {spatial_ndim} "
                    f"scale_factor = {scale_factor}, size = {size}"
                )
            torch_interp_mode = "linear"
    elif mode == "mask":
        torch_interp_mode = "area"
    elif mode == "segmentation":
        torch_interp_mode = "nearest-exact"

        if scale_factor is None:
            raise NotImplementedError()  # pragma: no cover
        if isinstance(scale_factor, float) and scale_factor < 1.0:
            raise NotImplementedError()  # pragma: no cover
        if isinstance(scale_factor, list) and sum([i < 1.0 for i in scale_factor]):
            raise NotImplementedError()  # pragma: no cover
    else:
        torch_interp_mode = mode  # type: ignore # has to fit at this point

    return torch_interp_mode


@overload
def interpolate(
    data: npt.NDArray,
    size: Optional[Sequence[int]] = ...,
    scale_factor: Optional[Union[float, Sequence[float]]] = ...,
    mode: InterpolationMode = ...,
    mask_value_thr: float = ...,
    default_space_ndim: int = ...,
    allow_shape_rounding: bool = ...,
) -> npt.NDArray:  # pragma: no cover
    ...


@overload
def interpolate(
    data: torch.Tensor,
    size: Optional[Sequence[int]] = ...,
    scale_factor: Optional[Union[float, Sequence[float]]] = ...,
    mode: InterpolationMode = ...,
    mask_value_thr: float = ...,
    default_space_ndim: int = ...,
    allow_shape_rounding: bool = ...,
) -> torch.Tensor:  # pragma: no cover
    ...


@typechecked
def interpolate(
    data: zu.typing.Tensor,
    size: Optional[Sequence[int]] = None,
    scale_factor: Optional[Union[float, Sequence[float]]] = None,
    mode: InterpolationMode = "img",
    mask_value_thr: float = 0,
    default_space_ndim: int = 2,
    allow_shape_rounding: bool = False,
):
    scale_factor = _standardize_scale_factor(
        data_ndim=data.ndim,
        scale_factor=scale_factor,
        default_space_ndim=default_space_ndim,
    )
    spatial_ndim = _get_spatial_ndim(
        size=size,
        scale_factor=scale_factor,
        data_shape=data.shape,
        allow_shape_rounding=allow_shape_rounding,
    )
    torch_interp_mode = _get_torch_interp_mode(
        size=size,
        scale_factor=scale_factor,
        spatial_ndim=spatial_ndim,
        mode=mode,
    )

    data_in = zu.tensor.convert.to_torch(data).float()
    result_raw = torch.nn.functional.interpolate(
        data_in,
        size=size,
        scale_factor=scale_factor,
        mode=torch_interp_mode,
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
            # if result_raw.shape[1] != spatial_ndim:
            #     raise ValueError(
            #         f"Malformed field shape: {result_raw.shape}. Number "
            #         f"of channels ({result_raw.shape[1]}) must be equal to "
            #         f"the number of spatial dimensions ({spatial_ndim})."
            #     )

        result_raw *= multiplier
    elif mode == "mask":
        result_raw = result_raw > mask_value_thr
    elif mode == "segmentation":
        result_raw = result_raw.int()

    result = zu.tensor.convert.astype(result_raw, data)
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


@overload
def compare(
    data: npt.NDArray,
    mode: CompareMode,
    operand: float,
    binarize: bool = ...,
    fill: Optional[float] = ...,
) -> npt.NDArray:  # pragma: no cover
    ...


@overload
def compare(
    data: torch.Tensor,
    mode: CompareMode,
    operand: float,
    binarize: bool = ...,
    fill: Optional[float] = ...,
) -> torch.Tensor:  # pragma: no cover
    ...


@typechecked
def compare(
    data: Tensor,
    mode: CompareMode,
    operand: float,
    binarize: bool = True,
    fill: Optional[float] = None,
) -> Tensor:
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


MaskFilteringModes = Literal["keep_large", "keep_small"]


@overload
def filter_cc(
    data: torch.Tensor,
    mode: MaskFilteringModes = ...,
    thr: int = ...,
) -> torch.Tensor:  # pragma: no cover
    ...


@overload
def filter_cc(
    data: npt.NDArray,
    mode: MaskFilteringModes = ...,
    thr: int = ...,
) -> npt.NDArray:  # pragma: no cover
    ...


@typechecked
def filter_cc(
    data: zu.typing.Tensor,
    mode: MaskFilteringModes = "keep_small",
    thr: int = 100,
) -> zu.typing.Tensor:
    """
    Remove connected components from the given input tensor.

    Clustering is performed based on non-zero values.

    Args:
        data (zu.typing.Tensor): Input tensor.
        mode (Literal["keep_large", "keep_small"]): Filtering mode.
        thr (int): Pixel size threshold.

    Returns:
        zu.typing.Tensor: Tensor with the filtered clusters removed.
    """
    data_np = zu.tensor.convert.to_np(data)
    cc_labels = cc3d.connected_components(data_np != 0)
    segids, counts = np.unique(cc_labels, return_counts=True)
    if mode == "keep_large":
        segids = [segid for segid, ct in zip(segids, counts) if ct > thr]
    else:
        segids = [segid for segid, ct in zip(segids, counts) if ct <= thr]

    filtered_mask = fastremap.mask_except(cc_labels, segids, in_place=True) != 0

    result_raw = copy.copy(data_np)
    result_raw[filtered_mask == 0] = 0

    result = zu.tensor.convert.astype(result_raw, data)
    return result

# pylint: disable=missing-docstring
from typing import Union, Literal, Optional, overload, Sequence, SupportsIndex
import numpy as np
import numpy.typing as npt
import torch
import einops  # type: ignore
from typeguard import typechecked

import zetta_utils as zu
from zetta_utils.typing import Tensor, Number

zu.builder.register("einops.rearrange")(einops.rearrange)
zu.builder.register("einpos.reduce")(einops.reduce)
zu.builder.register("einpos.repeat")(einops.repeat)


@zu.builder.register("multiply")
def multiply(data: Tensor, x) -> Tensor:  # pragma: no cover
    return x * data


@zu.builder.register("add")
def add(data: Tensor, x) -> Tensor:  # pragma: no cover
    return x + data


@zu.builder.register("power")
def power(data: Tensor, x) -> Tensor:  # pragma: no cover
    return data ** x


@zu.builder.register("divide")
def divide(data: Tensor, x) -> Tensor:  # pragma: no cover
    return data / x


@zu.builder.register("int_divide")
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


@zu.builder.register("unsqueeze")
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


@zu.builder.register("squeeze")
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
    scale_factor: Optional[Union[Number, Sequence[Number]]] = None,
) -> Optional[Sequence[float]]:
    if scale_factor is None:
        result = None
    else:
        data_space_ndim = data_ndim - 2  # Batch + Channel
        if isinstance(scale_factor, (float, int)):
            result = (scale_factor,) * data_space_ndim
        else:
            result = tuple(scale_factor)
            while len(result) < data_space_ndim:
                result = (1.0,) + result

    return result


def _get_torch_interp_mode(
    scale_factor_tuple: Optional[Sequence[float]],
    spatial_ndim: int,
    mode: InterpolationMode,
) -> TorchInterpolationMode:
    if mode in ("img", "field"):
        if spatial_ndim == 3:
            torch_interp_mode = "trilinear"  # type: TorchInterpolationMode
        elif spatial_ndim == 2:
            torch_interp_mode = "bilinear"
        else:
            assert spatial_ndim == 1, "Setting validation error."

            torch_interp_mode = "linear"
    elif mode == "mask":
        torch_interp_mode = "area"
    elif mode == "segmentation":
        torch_interp_mode = "nearest-exact"

        if scale_factor_tuple is None:
            raise NotImplementedError()
        if sum([i < 1.0 for i in scale_factor_tuple]):
            raise NotImplementedError()
    else:
        torch_interp_mode = mode  # type: ignore # has to fit at this point

    return torch_interp_mode


def _validate_interpolation_setting(
    data: zu.typing.Tensor,
    size: Optional[Sequence[int]],
    scale_factor_tuple: Optional[Sequence[float]],
    allow_shape_rounding: bool,
):
    # Torch checks for some of these, but we need to check preemptively
    # as some of our pre-processing code assumes a valid setting.

    if data.ndim > 5:
        raise ValueError(f"Number of dimensions must be <= 5. Got: {data.ndim}")

    if scale_factor_tuple is None and size is None:
        raise ValueError("Neither `size` nor `scale_factor` provided to `interpolate()`")
    if scale_factor_tuple is not None and size is not None:
        raise ValueError(
            "Both `size` and `scale_factor` provided to `interpolate()`. "
            "Exactly one of them must be provided."
        )

    spatial_ndim = data.ndim - 2
    if size is not None:
        if len(size) != spatial_ndim:
            raise ValueError(
                "`len(size)` must be equal to `data.ndim - 2`. "
                f"Got `len(size)` == {len(size)},  `data.ndim` == {data.ndim}."
            )

    if scale_factor_tuple is not None:
        if not allow_shape_rounding:
            result_spatial_shape = [
                data.shape[2 + i] * scale_factor_tuple[i] for i in range(spatial_ndim)
            ]
            for i in range(spatial_ndim):
                if round(result_spatial_shape[i]) != result_spatial_shape[i]:
                    raise RuntimeError(
                        f"Interpolation of array with shape {data.shape} and scale "
                        f"factor {scale_factor_tuple} would result in a non-integer shape "
                        f"along spatial dimention {i} "
                        f"({data.shape[2 + i]} -> {result_spatial_shape[i]}) while "
                        "`allow_shape_rounding` == False ."
                    )


@overload
def interpolate(
    data: npt.NDArray,
    size: Optional[Sequence[int]] = ...,
    scale_factor: Optional[Union[float, Sequence[float]]] = ...,
    mode: InterpolationMode = ...,
    mask_value_thr: float = ...,
    allow_shape_rounding: bool = ...,
    unsqueeze_to: Optional[int] = ...,
) -> npt.NDArray:  # pragma: no cover
    ...


@overload
def interpolate(
    data: torch.Tensor,
    size: Optional[Sequence[int]] = ...,
    scale_factor: Optional[Union[float, Sequence[float]]] = ...,
    mode: InterpolationMode = ...,
    mask_value_thr: float = ...,
    allow_shape_rounding: bool = ...,
    unsqueeze_to: Optional[int] = ...,
) -> torch.Tensor:  # pragma: no cover
    ...


@zu.builder.register("interpolate")
@typechecked
def interpolate(  # pylint: disable=too-many-locals
    data: zu.typing.Tensor,
    size: Optional[Sequence[int]] = None,
    scale_factor: Optional[Union[float, Sequence[float]]] = None,
    mode: InterpolationMode = "img",
    mask_value_thr: float = 0,
    allow_shape_rounding: bool = False,
    unsqueeze_to: Optional[int] = None,
) -> zu.typing.Tensor:
    """Interpolate the given tensor to the given ``size`` or by the given ``scale_factor``.

    :param data: Input tensor with batch and channel dimensions.
    :param size: Desired result shape.
    :param scale_factor: Interpolation scale factor.
        When provided as ``float``, applied to all spatial dimensions of the data.
    :param mode: Algorithm according to which the tensor should be interpolated.
    :param mask_value_thr: When ``mode == 'mask'``, threshold above which the interpolated
        value will be considered as ``True``.
    :param allow_shape_rounding: Whether to allow interpolation with scale factors that
        result in non-integer tensor shapes.
    :param unsqueeze_to: If provided, the tensor will be unsqueezed to the given number
        of dimensions before interpolating. New dimensions are alwyas added to the front
        (dim 0). Result is squeezed back to the original number of dimensions before
        returning.
    :return: Interpolated tensor of the same type as the input tensor_ops.

    """
    unsqueeze_count = 0
    if unsqueeze_to is not None:
        while data.ndim < unsqueeze_to:
            data = unsqueeze(data, 0)
            unsqueeze_count += 1

    scale_factor_tuple = _standardize_scale_factor(
        data_ndim=data.ndim,
        scale_factor=scale_factor,
    )

    _validate_interpolation_setting(
        data=data,
        size=size,
        scale_factor_tuple=scale_factor_tuple,
        allow_shape_rounding=allow_shape_rounding,
    )

    torch_interp_mode = _get_torch_interp_mode(
        scale_factor_tuple=scale_factor_tuple,
        spatial_ndim=data.ndim - 2,
        mode=mode,
    )

    data_in = zu.tensor_ops.convert.to_torch(data).float()
    result_raw = torch.nn.functional.interpolate(
        data_in,
        size=size,
        scale_factor=scale_factor_tuple,
        mode=torch_interp_mode,
    )

    if mode == "field":
        if scale_factor_tuple is None:
            raise NotImplementedError(  # pragma: no cover
                "`size`-based field interpolation is not currently supported."
            )
        if all(e == scale_factor_tuple[0] for e in scale_factor_tuple):
            multiplier = scale_factor_tuple[0]
        else:
            raise NotImplementedError(  # pragma: no cover
                f"Non-isotropic field interpolation (scale_factor={scale_factor_tuple}) "
                "is not currently supported."
            )

        result_raw *= multiplier
    elif mode == "mask":
        result_raw = result_raw > mask_value_thr
    elif mode == "segmentation":
        result_raw = result_raw.int()

    result = zu.tensor_ops.convert.astype(result_raw, data)

    for _ in range(unsqueeze_count):
        result = squeeze(result, 0)

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


@zu.builder.register("compare")
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

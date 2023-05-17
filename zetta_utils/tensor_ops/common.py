# pylint: disable=missing-docstring
from typing import Callable, Literal, Optional, Sequence, SupportsIndex, Union

import einops
import numpy as np
import torch
from typeguard import typechecked

from zetta_utils import builder, tensor_ops
from zetta_utils.tensor_typing import Tensor, TensorTypeVar


@builder.register("rearrange")
def rearrange(data: TensorTypeVar, **kwargs) -> TensorTypeVar:  # pragma: no cover
    return einops.rearrange(tensor=data, **kwargs)  # type: ignore # bad typing by einops


@builder.register("reduce")
def reduce(data: TensorTypeVar, **kwargs) -> TensorTypeVar:  # pragma: no cover
    return einops.reduce(tensor=data, **kwargs)


@builder.register("repeat")
def repeat(data: TensorTypeVar, **kwargs) -> TensorTypeVar:  # pragma: no cover
    return einops.repeat(tensor=data, **kwargs)


@builder.register("multiply")
def multiply(data: TensorTypeVar, value) -> TensorTypeVar:  # pragma: no cover
    return value * data


@builder.register("add")
def add(data: TensorTypeVar, value) -> TensorTypeVar:  # pragma: no cover
    return value + data


@builder.register("power")
def power(data: TensorTypeVar, value) -> TensorTypeVar:  # pragma: no cover
    return data ** value


@builder.register("divide")
def divide(data: TensorTypeVar, value) -> TensorTypeVar:  # pragma: no cover
    return data / value


@builder.register("int_divide")
def int_divide(data: TensorTypeVar, value) -> TensorTypeVar:  # pragma: no cover
    return data // value


@builder.register("unsqueeze")
@typechecked
def unsqueeze(
    data: TensorTypeVar, dim: Union[SupportsIndex, Sequence[SupportsIndex]] = 0
) -> TensorTypeVar:
    """
    Returns a new tensor with new dimensions of size one inserted at the specified positions.

    :param data: the input tensor.
    :param dim: indexes at which to insert new dimensions.
    :return: tensor with added dimensions.
    """
    if isinstance(data, torch.Tensor):
        if isinstance(dim, int):
            result = data.unsqueeze(dim)  # type: TensorTypeVar
        else:
            raise ValueError(f"Cannot use `torch.unsqueeze` with dim of type '{type(dim)}'")
    else:
        assert isinstance(data, np.ndarray), "Type checking failure"
        result = np.expand_dims(data, dim)

    return result


@builder.register("squeeze")
@typechecked
def squeeze(
    data: TensorTypeVar, dim: Optional[Union[SupportsIndex, Sequence[SupportsIndex]]] = None
) -> TensorTypeVar:
    """
    Returns a tensor with all the dimensions of input of size 1 removed.
    When dim is given, a squeeze operation is done only for the given dimensions.

    :param data: the input tensor.
    :param dim:  if given, the input will be squeezed only in these dimensions.
    :return: tensor with squeezed dimensions.
    """

    if isinstance(data, torch.Tensor):
        if isinstance(dim, int) or dim is None:
            result = data.squeeze(dim)
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
    data: Tensor,
    size: Optional[Sequence[int]],
    scale_factor_tuple: Optional[Sequence[float]],
    allow_slice_rounding: bool,
):
    # Torch checks for some of these, but we need to check preemptively
    # as some of our pre-processing code assumes a valid setting.

    if data.ndim > 5:
        raise ValueError(f"float of dimensions must be <= 5. Got: {data.ndim}")

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
        if not allow_slice_rounding:
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
                        "`allow_slice_rounding` == False ."
                    )


@builder.register("unsqueeze_to")
@typechecked
def unsqueeze_to(
    data: TensorTypeVar,
    ndim: Optional[int],
):
    """
    Returns a new tensor with new dimensions of size one inserted at poisition 0 until
    the tensor reaches the given number of dimensions.

    :param data: the input tensor.
    :param ndim: target number of dimensions.
    :return: tensor with added dimensions.
    """

    if ndim is not None:
        while data.ndim < ndim:
            data = unsqueeze(data, 0)
    return data


@builder.register("squeeze_to")
@typechecked
def squeeze_to(
    data: TensorTypeVar,
    ndim: Optional[int],
):
    """
    Returns a new tensor with the dimension at position 0 squeezed until
    the tensor reaches the given number of dimensions.

    :param data: the input tensor.
    :param ndim: target number of dimensions.
    :return: tensor with squeezed dimensions.
    """
    if ndim is not None:
        while data.ndim > ndim:
            if data.shape[0] != 1:
                raise RuntimeError(
                    f"Not able to squeeze tensor with shape=={data.shape} to "
                    f"ndim=={ndim}: shape[0] != 1"
                )
            data = squeeze(data, 0)

    return data


@builder.register("interpolate")
@typechecked
def interpolate(  # pylint: disable=too-many-locals
    data: TensorTypeVar,
    size: Optional[Sequence[int]] = None,
    scale_factor: Optional[Union[float, Sequence[float]]] = None,
    mode: InterpolationMode = "img",
    mask_value_thr: float = 0,
    allow_slice_rounding: bool = False,
    unsqueeze_input_to: Optional[int] = 5,
) -> TensorTypeVar:
    """
    Interpolate the given tensor to the given ``size`` or by the given ``scale_factor``.

    :param data: Input tensor with batch and channel dimensions.
    :param size: Desired result shape.
    :param scale_factor: Interpolation scale factor.
        When provided as ``float``, applied to all spatial dimensions of the data.
    :param mode: Algorithm according to which the tensor should be interpolated.
    :param mask_value_thr: When ``mode == 'mask'``, threshold above which the interpolated
        value will be considered as ``True``.
    :param allow_slice_rounding: Whether to allow interpolation with scale factors that
        result in non-integer tensor shapes.
    :param unsqueeze_to: If provided, the tensor will be unsqueezed to the given number
        of dimensions before interpolating. New dimensions are alwyas added to the front
        (dim 0). Result is squeezed back to the original number of dimensions before
        returning.
    :return: Interpolated tensor of the same type as the input tensor_ops.
    """
    original_ndim = data.ndim
    # breakpoint()
    data = unsqueeze_to(data, unsqueeze_input_to)

    scale_factor_tuple = _standardize_scale_factor(
        data_ndim=data.ndim,
        scale_factor=scale_factor,
    )

    _validate_interpolation_setting(
        data=data,
        size=size,
        scale_factor_tuple=scale_factor_tuple,
        allow_slice_rounding=allow_slice_rounding,
    )

    torch_interp_mode = _get_torch_interp_mode(
        scale_factor_tuple=scale_factor_tuple,
        spatial_ndim=data.ndim - 2,
        mode=mode,
    )
    data_torch = tensor_ops.convert.to_torch(data)
    data_in = data_torch.float()
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
        field_dim = result_raw.shape[1]
        if field_dim == 2:
            if scale_factor_tuple[0] != scale_factor_tuple[1]:
                raise NotImplementedError(  # pragma: no cover
                    f"Non-isotropic 2D field interpolation is not supported. "
                    f"X scale factor: {scale_factor_tuple[0]} "
                    f"y scale factor: {scale_factor_tuple[1]} "
                )
            multiplier = scale_factor_tuple[0]
        else:
            raise NotImplementedError("Only 2D field interpolation is currently supported")
        # All of the field dimensions
        if all(e == scale_factor_tuple[0] for e in scale_factor_tuple[:field_dim]):
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

    result = result_raw.to(data_torch.dtype)
    result = tensor_ops.convert.astype(result, data)
    result = squeeze_to(result, original_ndim)

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


@builder.register("compare")
@typechecked
def compare(
    data: TensorTypeVar,
    mode: CompareMode,
    value: float,
    binarize: bool = True,
    fill: Optional[float] = None,
) -> TensorTypeVar:
    """
    Compare the given the given tensor to the given value.
    If `binarize` is set to `True`, return the binary outcome of the comparison.
    If `fill` is set, return a new tensor in which the values that pass the comparison
    are replaced by `fill`. Only one of `binarize=True` or `fill` can be set.

    :param data: the input tensor.
    :param mode: the mode of comparison.
    :param value: comparison operand.
    :param binarize: when set to `True`, will return a binary result of comparison, and `fill` must
    be `None`.
    :param fill: when set, will return a new tensor with values that pass the comparison
    replaced by `fill`, and `binarize` must be `False`.
    """

    if mode in ["eq", "=="]:
        mask = data == value
    elif mode in ["neq", "!="]:
        mask = data != value
    elif mode in ["gt", ">"]:
        mask = data > value
    elif mode in ["gte", ">="]:
        mask = data >= value
    elif mode in ["lt", "<"]:
        mask = data < value
    elif mode in ["lte", "<="]:
        mask = data <= value
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


@builder.register("crop")
@typechecked
def crop(
    data: TensorTypeVar,
    crop: Sequence[int],  # pylint: disable=redefined-outer-name
    # mode: Literal["center"] = "center",
) -> TensorTypeVar:
    """
    Crop a multidimensional tensor.

    :param data: Input tensor.
    :param crop: float from pixels to crop from each side.
        The last integer will correspond to the last dimension, and count
        will go from there right to left.
    """

    slices = [slice(0, None) for _ in range(data.ndim - len(crop))]
    for e in crop:
        assert e >= 0
        if e != 0:
            slices.append(slice(e, -e))
        else:
            slices.append(slice(0, None))
    result = data[tuple(slices)]
    return result


@builder.register("split_reduce")
@typechecked
def split_reduce(
    data: TensorTypeVar,
    paths: list[Callable[[TensorTypeVar], TensorTypeVar]],
    reduce_mode: Literal["maximum"] = "maximum",
) -> TensorTypeVar:
    """
    :param data: the input tensor.
    :param dim:  if given, the input will be squeezed only in these dimensions.
    :return: tensor with squeezed dimensions.
    """
    path_outcomes = [e(data) for e in paths]
    assert reduce_mode == "maximum"
    result_torch = tensor_ops.convert.to_torch(path_outcomes[0])
    for e in path_outcomes[1:]:
        result_torch = torch.maximum(result_torch, tensor_ops.convert.to_torch(e))
    result = tensor_ops.convert.astype(result_torch, data)

    return result

"""Type conversion functions."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from typeguard import typechecked

from zetta_utils import builder, tensor_ops
from zetta_utils.tensor_ops.common import supports_dict
from zetta_utils.tensor_typing import Tensor, TensorTypeVar


@typechecked
def to_np(data: Tensor) -> npt.NDArray:
    """Convert the given tensor to ``numpy.ndarray``.

    :param data: Input tensor_ops.
    :return: Input tensor in ``numpy.ndarray`` format.

    """
    if isinstance(data, torch.Tensor):
        result = data.cpu().detach().numpy()
    else:
        assert isinstance(data, np.ndarray)
        result = data

    return result


@typechecked
def to_torch(data: Tensor, device: torch.types.Device = None) -> torch.Tensor:
    """Convert the given tensor to `torch.Tensor`.

    :param data: Input tensor_ops.
    :param device: Device name on which the torch tensor will reside.
    :return: Input tensor in `torch.Tensor` format.

    """
    if isinstance(data, torch.Tensor):
        result = data.to(device=device)  # type: ignore # pytorch bug
    else:
        assert isinstance(data, np.ndarray)
        if data.dtype == np.uint64:
            if data.max() > np.uint64(2 ** 63 - 1):
                raise ValueError("Unable to convert uint64 dtype to int64")
            data = data.astype(np.int64)
        elif data.dtype == np.uint32:
            if data.max() > np.uint32(2 ** 31 - 1):
                raise ValueError("Unable to convert uint32 dtype to int32")
            data = data.astype(np.int32)

        if any(v < 0 for v in data.strides):  # torch.from_numpy does not support negative strides
            data = data.copy("K")
        result = torch.from_numpy(data).to(device)  # type: ignore # pytorch bug

    return result


@typechecked
def astype(
    data: Tensor,
    reference: TensorTypeVar,
    cast: bool = False,
) -> TensorTypeVar:
    """Convert the given tensor to `np.ndarray` or `torch.Tensor`
    depending on the type of reference tensor_ops.

    :param data: Input tensor_ops.
    :param reference: Reference type tensor_ops.
    :param cast: If ``True``, cast `data` to the type of `reference`.
    :return: Input tensor converted to the reference type.

    """
    if isinstance(reference, torch.Tensor):
        data_tc = tensor_ops.convert.to_torch(data)
        return data_tc.to(reference.dtype) if cast else data_tc
    else:
        assert isinstance(reference, np.ndarray)
        data_np = tensor_ops.convert.to_np(data)
        return data_np.astype(reference.dtype) if cast else data_np


@builder.register("to_float32")
@typechecked
@supports_dict
def to_float32(data: TensorTypeVar) -> TensorTypeVar:
    """Convert the given tensor to fp32.

    :param data: Input tensor_ops.
    :return: Input tensor converted to fp32.

    """
    if isinstance(data, torch.Tensor):
        result = data.float()  # type: TensorTypeVar
    elif isinstance(data, np.ndarray):
        result = data.astype(np.float32)

    return result


@builder.register("to_uint8")
@typechecked
@supports_dict
def to_uint8(data: TensorTypeVar) -> TensorTypeVar:
    """Convert the given tensor to uint8.

    :param data: Input tensor_ops.
    :return: Input tensor converted to uint8.

    """
    if isinstance(data, torch.Tensor):
        result = data.byte()  # type: TensorTypeVar
    elif isinstance(data, np.ndarray):
        result = data.astype(np.uint8)

    return result

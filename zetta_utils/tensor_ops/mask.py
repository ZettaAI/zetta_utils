import copy
from typing import Literal, Protocol, TypeVar

import cc3d
import einops
import fastremap
import numpy as np
import scipy
import torch
from typeguard import typechecked
from typing_extensions import ParamSpec

from zetta_utils import builder
from zetta_utils.tensor_typing import TensorTypeVar

from . import convert

MaskFilteringModes = Literal["keep_large", "keep_small"]

P = ParamSpec("P")


class TensorOp(Protocol[P]):
    """
    Protocol which defines what it means for a funciton to be a tensor_op:
    it must take a `data` argument of TensorTypeVar type, and return a
    tensor of the same type.
    """

    def __call__(self, data: TensorTypeVar, *args: P.args, **k: P.kwargs) -> TensorTypeVar:
        ...


OpT = TypeVar("OpT", bound=TensorOp)


def skip_on_empty_data(fn: OpT) -> OpT:
    """
    Decorator that ensures early exit for a tensor op when `data` is 0.
    """

    def wrapped(data: TensorTypeVar, *args: P.args, **kwargs: P.kwargs) -> TensorTypeVar:
        if (data != 0).sum() == 0:
            result = data
        else:
            result = fn(data, *args, **kwargs)
        return result

    return wrapped  # type: ignore


@builder.register("filter_cc")  # type: ignore # TODO: pyright
@skip_on_empty_data
@typechecked
def filter_cc(
    data: TensorTypeVar,
    mode: MaskFilteringModes = "keep_small",
    thr: int = 100,
) -> TensorTypeVar:
    """
    Remove connected components from the given input tensor_ops.

    Clustering is performed based on non-zero values.

    :param data: Input tensor (CXYZ).
    :param mode: Filtering mode.
    :param thr:  Pixel size threshold.
    :return: Tensor with the filtered clusters removed.
    """
    data_np = convert.to_np(data)

    data_np = einops.rearrange(data_np, "C X Y Z -> Z C X Y")
    assert data_np.shape[1] == 1

    result_raw = np.zeros_like(data_np)

    for z in range(data_np.shape[0]):
        if (data_np[z] != 0).sum() > 0:
            cc_labels = cc3d.connected_components(data_np[z] != 0)
            segids, counts = np.unique(cc_labels, return_counts=True)
            if mode == "keep_large":
                segids = [segid for segid, ct in zip(segids, counts) if ct > thr]
            else:
                segids = [segid for segid, ct in zip(segids, counts) if ct <= thr]

            filtered_mask = fastremap.mask_except(cc_labels, segids, in_place=True) != 0

            result_raw[z] = copy.copy(data_np[z])
            result_raw[z][filtered_mask == 0] = 0

    result_raw = einops.rearrange(result_raw, "Z C X Y -> C X Y Z")
    result = convert.astype(result_raw, data)
    return result


@builder.register("coarsen")  # type: ignore
@skip_on_empty_data
@typechecked
def coarsen(data: TensorTypeVar, width: int = 1, thr: int = 3) -> TensorTypeVar:
    """
    Coarsen the given mask.

    :param data: Input mask tensor (CXYZ).
    :param width: Amount of pixels by which to coarsen.
    :return: Coarsened mask tensor.
    """
    data_torch_cxyz = convert.to_torch(data)

    data_torch = einops.rearrange(data_torch_cxyz, "C X Y Z -> Z C X Y")

    kernel = torch.ones(
        [1, 1]
        + [
            3,
        ]
        * (data_torch.ndim - 2),
        device=data_torch.device,
    )
    result_torch = data_torch.float()
    for _ in range(width):
        conved = torch.nn.functional.conv2d(result_torch, kernel, padding=1)
        result_torch = (conved >= thr).float()

    result_torch = (result_torch > 0).to(data_torch.dtype)

    result = convert.astype(einops.rearrange(result_torch, "Z C X Y -> C X Y Z"), data)
    return result


@builder.register("erode")  # type: ignore
@skip_on_empty_data
@typechecked
def erode(data: TensorTypeVar, width: int = 1, thr: int = 3) -> TensorTypeVar:
    """
    Erode the given mask.

    :param data: Input mask tensor (CXYZ).
    :param width: Amount of pixels by which to erode.
    :return: Eroded mask tensor.
    """
    result = coarsen(data == 0, thr=thr, width=width) == 0
    return result


@builder.register("binary_closing")  # type: ignore
@skip_on_empty_data
@typechecked
def binary_closing(data: TensorTypeVar, width: int = 1) -> TensorTypeVar:
    """
    Run binary closing on the mask.

    :param data: Input mask tensor (CXYZ).
    :param width: Number of closing iterations.
    :return: Closed mask tensor.
    """
    data_np = convert.to_np(data)
    result_np = np.ones_like(data_np)

    # CXYZ
    assert len(data.shape) == 4
    assert data.shape[0] == 1

    for i in range(data_np.shape[-1]):
        slice_data = data_np[0, :, :, i]
        slice_result = scipy.ndimage.binary_closing(slice_data, iterations=width)
        # Prevent boundary erosion
        slice_result[:width, :] |= slice_data[:width, :].astype(np.bool_)
        slice_result[-width:, :] |= slice_data[-width:, :].astype(np.bool_)
        slice_result[:, :width] |= slice_data[:, :width].astype(np.bool_)
        slice_result[:, -width:] |= slice_data[:, -width:].astype(np.bool_)
        slice_result = slice_result.astype(data_np.dtype)

        result_np[0, :, :, i] = slice_result

    result = convert.astype(result_np, data) > 0
    return result

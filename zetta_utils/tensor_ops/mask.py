import copy
from typing import Literal, Protocol, TypeVar

import cc3d
import einops
import fastremap
import numpy as np
import scipy
import torch
from kornia import morphology
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


@builder.register("coarsen_mask")  # type: ignore # TODO: pyright
@skip_on_empty_data
@typechecked
def coarsen(data: TensorTypeVar, width: int = 1, thr: int = 1) -> TensorTypeVar:
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


@builder.register("binary_closing")  # type: ignore
@skip_on_empty_data
@typechecked
def binary_closing(data: TensorTypeVar, iterations: int = 1) -> TensorTypeVar:
    """
    Run binary closing on the mask.

    :param data: Input mask tensor (CXYZ).
    :param iterations: Number of closing iterations.
    :return: Closed mask tensor.
    """
    data_np = convert.to_np(data)

    if len(data.shape) == 4:
        # CXYZ
        assert data.shape[0] == 1
        assert data.shape[-1] == 1
        data_np = data_np.squeeze(0).squeeze(-1)

    result_raw = scipy.ndimage.binary_closing(data_np, iterations=iterations)
    # Prevent boundary erosion
    result_raw[..., :iterations, :] |= data_np[..., :iterations, :].astype(np.bool_)
    result_raw[..., -iterations:, :] |= data_np[..., -iterations:, :].astype(np.bool_)
    result_raw[..., :, :iterations] |= data_np[..., :, :iterations].astype(np.bool_)
    result_raw[..., :, -iterations:] |= data_np[..., :, -iterations:].astype(np.bool_)
    result_raw = result_raw.astype(data_np.dtype)

    if len(data.shape) == 4:
        result_raw = np.expand_dims(result_raw, (0, -1))

    result = convert.astype(result_raw, data) > 0
    return result


@builder.register("kornia_opening")  # type: ignore
@skip_on_empty_data
@typechecked
def kornia_opening(
    data: TensorTypeVar,
    kernel: TensorTypeVar,
    device: torch.types.Device = None,
    **kwargs,
) -> TensorTypeVar:
    data_torch_cxyz = convert.to_torch(data, device=device)
    kernel_torch = convert.to_torch(kernel, device=data_torch_cxyz.device)
    data_torch = einops.rearrange(data_torch_cxyz, "C X Y Z -> Z C X Y")

    result_torch = morphology.opening(data_torch.float(), kernel=kernel_torch, **kwargs)

    result_torch = result_torch.to(data_torch.dtype)
    result = convert.astype(einops.rearrange(result_torch, "Z C X Y -> C X Y Z"), data)
    return result


@builder.register("kornia_closing")  # type: ignore
@skip_on_empty_data
@typechecked
def kornia_closing(
    data: TensorTypeVar,
    kernel: TensorTypeVar,
    device: torch.types.Device = None,
    **kwargs,
) -> TensorTypeVar:
    data_torch_cxyz = convert.to_torch(data, device=device)
    kernel_torch = convert.to_torch(kernel, device=data_torch_cxyz.device)
    data_torch = einops.rearrange(data_torch_cxyz, "C X Y Z -> Z C X Y")

    result_torch = morphology.closing(data_torch.float(), kernel=kernel_torch, **kwargs)

    result_torch = result_torch.to(data_torch.dtype)
    result = convert.astype(einops.rearrange(result_torch, "Z C X Y -> C X Y Z"), data)
    return result


@builder.register("kornia_erosion")  # type: ignore
@skip_on_empty_data
@typechecked
def kornia_erosion(
    data: TensorTypeVar,
    kernel: TensorTypeVar,
    device: torch.types.Device = None,
    **kwargs,
) -> TensorTypeVar:
    data_torch_cxyz = convert.to_torch(data, device=device)
    kernel_torch = convert.to_torch(kernel, device=data_torch_cxyz.device)
    data_torch = einops.rearrange(data_torch_cxyz, "C X Y Z -> Z C X Y")

    result_torch = morphology.erosion(data_torch.float(), kernel=kernel_torch, **kwargs)

    result_torch = result_torch.to(data_torch.dtype)
    result = convert.astype(einops.rearrange(result_torch, "Z C X Y -> C X Y Z"), data)
    return result


@builder.register("kornia_dilation")  # type: ignore
@skip_on_empty_data
@typechecked
def kornia_dilation(
    data: TensorTypeVar,
    kernel: TensorTypeVar,
    device: torch.types.Device = None,
    **kwargs,
) -> TensorTypeVar:
    data_torch_cxyz = convert.to_torch(data, device=device)
    kernel_torch = convert.to_torch(kernel, device=data_torch_cxyz.device)
    data_torch = einops.rearrange(data_torch_cxyz, "C X Y Z -> Z C X Y")

    result_torch = morphology.dilation(data_torch.float(), kernel=kernel_torch, **kwargs)

    result_torch = result_torch.to(data_torch.dtype)
    result = convert.astype(einops.rearrange(result_torch, "Z C X Y -> C X Y Z"), data)
    return result

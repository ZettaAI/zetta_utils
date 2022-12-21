import copy
from typing import Literal

import cc3d
import fastremap
import numpy as np
import scipy
import torch
from typeguard import typechecked

from zetta_utils import builder, tensor_ops
from zetta_utils.tensor_typing import TensorTypeVar

from .common import skip_on_empty_data

MaskFilteringModes = Literal["keep_large", "keep_small"]


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
    data_np = tensor_ops.convert.to_np(data)

    if len(data.shape) == 4:
        assert data.shape[0] == 1
        assert data.shape[1] == 1
        data_np = data_np.squeeze(0).squeeze(0)

    cc_labels = cc3d.connected_components(data_np != 0)
    segids, counts = np.unique(cc_labels, return_counts=True)
    if mode == "keep_large":
        segids = [segid for segid, ct in zip(segids, counts) if ct > thr]
    else:
        segids = [segid for segid, ct in zip(segids, counts) if ct <= thr]

    filtered_mask = fastremap.mask_except(cc_labels, segids, in_place=True) != 0

    result_raw = copy.copy(data_np)
    result_raw[filtered_mask == 0] = 0

    if len(data.shape) == 4:
        result_raw = np.expand_dims(result_raw, (0, 1))

    result = tensor_ops.convert.astype(result_raw, data)

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

    data_torch = tensor_ops.convert.to_torch(data)
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
    result = tensor_ops.convert.astype(result_torch, data)
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
    data_np = tensor_ops.convert.to_np(data)

    if len(data.shape) == 4:
        assert data.shape[0] == 1
        assert data.shape[1] == 1
        data_np = data_np.squeeze(0).squeeze(0)

    result_raw = scipy.ndimage.binary_closing(data_np, iterations=iterations)
    # Prevent boundary erosion
    result_raw[..., :iterations, :] |= data_np[..., :iterations, :].astype(np.bool_)
    result_raw[..., -iterations:, :] |= data_np[..., -iterations:, :].astype(np.bool_)
    result_raw[..., :, :iterations] |= data_np[..., :, :iterations].astype(np.bool_)
    result_raw[..., :, -iterations:] |= data_np[..., :, -iterations:].astype(np.bool_)
    result_raw = result_raw.astype(data_np.dtype)

    if len(data.shape) == 4:
        result_raw = np.expand_dims(result_raw, (0, 1))

    result = tensor_ops.convert.astype(result_raw, data)

    return result

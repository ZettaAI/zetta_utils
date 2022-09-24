from typing import Literal
import copy
import torch
import numpy as np
from typeguard import typechecked

import cc3d  # type: ignore
import fastremap  # type: ignore

from zetta_utils import builder, tensor_ops
from zetta_utils.tensor_typing import TensorTypeVar


MaskFilteringModes = Literal["keep_large", "keep_small"]


@builder.register("filter_cc")
@typechecked
def filter_cc(
    data: TensorTypeVar,
    mode: MaskFilteringModes = "keep_small",
    thr: int = 100,
) -> TensorTypeVar:
    """
    Remove connected components from the given input tensor_ops.

    Clustering is performed based on non-zero values.

    :param data: Input tensor.
    :param mode: Filtering mode.
    :param thr:  Pixel size threshold.
    :return: Tensor with the filtered clusters removed.
    """
    data_np = tensor_ops.convert.to_np(data)
    cc_labels = cc3d.connected_components(data_np != 0)
    segids, counts = np.unique(cc_labels, return_counts=True)
    if mode == "keep_large":
        segids = [segid for segid, ct in zip(segids, counts) if ct > thr]
    else:
        segids = [segid for segid, ct in zip(segids, counts) if ct <= thr]

    filtered_mask = fastremap.mask_except(cc_labels, segids, in_place=True) != 0

    result_raw = copy.copy(data_np)
    result_raw[filtered_mask == 0] = 0

    result = tensor_ops.convert.astype(result_raw, data)
    return result


@builder.register("coarsen_mask")
@typechecked
def coarsen(data: TensorTypeVar, width: int = 1, thr: int = 1) -> TensorTypeVar:

    data_torch = tensor_ops.convert.to_torch(data).float()
    kernel = torch.ones(
        [1, 1]
        + [
            3,
        ]
        * (data_torch.ndim - 2),
        device=data_torch.device,
    )
    result_torch = data_torch
    for _ in range(width):
        conved = torch.nn.functional.conv2d(result_torch, kernel, padding=1)
        result_torch = (conved >= thr).float()

    result_torch = result_torch > 0
    result = tensor_ops.convert.astype(result_torch, data)
    return result

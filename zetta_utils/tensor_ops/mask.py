from typing import Literal, overload
import copy
import torch
import numpy as np
import numpy.typing as npt
from typeguard import typechecked

import cc3d  # type: ignore
import fastremap  # type: ignore
import zetta_utils as zu


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


@zu.builder.register("filter_cc")
@typechecked
def filter_cc(
    data: zu.typing.Tensor,
    mode: MaskFilteringModes = "keep_small",
    thr: int = 100,
) -> zu.typing.Tensor:
    """
    Remove connected components from the given input tensor_ops.

    Clustering is performed based on non-zero values.

    Args:
        data (zu.typing.Tensor): Input tensor_ops.
        mode (Literal["keep_large", "keep_small"]): Filtering mode.
        thr (int): Pixel size threshold.

    Returns:
        zu.typing.Tensor: Tensor with the filtered clusters removed.
    """
    data_np = zu.tensor_ops.convert.to_np(data)
    cc_labels = cc3d.connected_components(data_np != 0)
    segids, counts = np.unique(cc_labels, return_counts=True)
    if mode == "keep_large":
        segids = [segid for segid, ct in zip(segids, counts) if ct > thr]
    else:
        segids = [segid for segid, ct in zip(segids, counts) if ct <= thr]

    filtered_mask = fastremap.mask_except(cc_labels, segids, in_place=True) != 0

    result_raw = copy.copy(data_np)
    result_raw[filtered_mask == 0] = 0

    result = zu.tensor_ops.convert.astype(result_raw, data)
    return result


@overload
def coarsen(
    data: npt.NDArray, width: int = ..., thr: int = ...
) -> npt.NDArray:  # pragma: no cover
    ...


@overload
def coarsen(
    data: torch.Tensor, width: int = ..., thr: int = ...
) -> torch.Tensor:  # pragma: no cover
    ...


@zu.builder.register("coarsen_mask")
@typechecked
def coarsen(data: zu.typing.Tensor, width: int = 1, thr: int = 1) -> zu.typing.Tensor:

    data_torch = zu.tensor_ops.convert.to_torch(data).float()
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
    result = zu.tensor_ops.convert.astype(result_torch, data)
    return result

"""Tools for operating on binary masks."""
from __future__ import annotations

import copy
from typing import Literal
import numpy.typing as npt
import numpy as np
import fastremap  # type: ignore
import cc3d  # type: ignore
from typeguard import typechecked

MaskFilteringModes = Literal["keep_large", "keep_small"]


@typechecked
def filter_cc(
    a: npt.NDArray,
    mode: MaskFilteringModes = "keep_small",
    thr: int = 100,
):
    """
    Remove connected components from the given input array.

    Clustering is performed based on non-zero values.

    Args:
        a (npt.NDArray): Input array.
        mode (Literal["keep_large", "keep_small"]): Filtering mode.
        thr (int): Pixel size threshold.

    Returns:
        npt.NDArray: Input array with the filtered clusters removed.
    """
    cc_labels = cc3d.connected_components(a != 0)
    segids, counts = np.unique(cc_labels, return_counts=True)
    if mode == "keep_large":
        segids = [segid for segid, ct in zip(segids, counts) if ct > thr]
    else:
        segids = [segid for segid, ct in zip(segids, counts) if ct <= thr]

    filtered_mask = fastremap.mask_except(cc_labels, segids, in_place=True) != 0

    result = copy.copy(a)
    result[filtered_mask == 0] = 0

    return result

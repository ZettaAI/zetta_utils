from __future__ import annotations

from typing import Literal

import abiss
import einops
import numpy as np
import torch
from lsd.post.fragments import watershed_from_affinities as _watershed_lsd

from zetta_utils import builder


def _run_watershed_abiss(
    affs: torch.Tensor,
    aff_threshold_low: float = 0.01,
    aff_threshold_high: float = 0.99,
    size_threshold: int = 0,
    # agglomeration_threshold: float = 0.0,
) -> torch.Tensor:
    """
    Args:
        affs:
            Affinity tensor in float32 with values [0.0, 1.0].

        aff_threshold_low, aff_threshold_high:
            Low and high watershed thresholds.

        size_threshold:
            If greater than 0, perform single-linkage merging as a subsequent
            step.

        agglomeration_threshold:
            If greater than 0.0, perform agglomeration as a subsequent
            step with this threshold.
    """
    affs = torch.nn.functional.pad(affs, (1, 1, 1, 1, 1, 1))  # abiss requires 1px padding
    affs = einops.rearrange(affs, "C X Y Z -> X Y Z C")  # channel last
    ret = abiss.watershed(
        affs=affs.numpy(),
        aff_threshold_low=aff_threshold_low,
        aff_threshold_high=aff_threshold_high,
        size_threshold=size_threshold,
        # agglomeration_threshold=agglomeration_threshold,
    )
    ret = ret[1:-1, 1:-1, 1:-1]
    ret = np.expand_dims(ret, axis=0)
    return ret


def _run_watershed_lsd(
    affs: torch.Tensor,
    fragments_in_xy: bool = False,
    min_seed_distance: int = 10,
) -> torch.Tensor:
    """
    Args:
        affs:
            Affinity tensor in either float32 or uint8.

        fragments_in_xy:
            Produce supervoxels in xy.

        min_seed_distance:
            Controls distance between seeds in voxels.
    """
    """
    TODO:
    - add supervoxel filtering based on average aff value
    - add option to also perform agglomeration
    """
    affs = einops.rearrange(affs, "C X Y Z -> C Z Y X").numpy()
    if affs.dtype == np.uint8:
        max_affinity_value = 255.0
        affs = affs.astype(np.float32)  # type: ignore
    else:
        max_affinity_value = 1.0

    ret, _ = _watershed_lsd(
        affs=affs,
        max_affinity_value=max_affinity_value,
        fragments_in_xy=fragments_in_xy,
        min_seed_distance=min_seed_distance,
    )
    ret = einops.rearrange(ret, "Z Y X -> X Y Z")
    ret = np.expand_dims(ret, axis=0)
    return ret


@builder.register("watershed_from_affinities")
def watershed_from_affinities(
    affs: torch.Tensor,  # in CXYZ
    method: Literal["abiss", "lsd"],
    **kwargs,
) -> torch.Tensor:
    """
    Produce supervoxels by running watershed on aff data. Optionally perform
    agglomeration and output segmentation.
    """
    if method == "lsd":
        seg = _run_watershed_lsd(affs, **kwargs)
    elif method == "abiss":
        seg = _run_watershed_abiss(affs, **kwargs)
    """
    TODO: write a wrapper for multi-chunk watershed that performs:
        - relabel supervoxels based on chunkid & chunk size
        - add supervoxel filtering based on mask
        - store a list of supervoxels within a chunk to a database
    """
    return seg

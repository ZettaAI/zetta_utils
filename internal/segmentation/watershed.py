from __future__ import annotations

from typing import Literal

import abiss
import einops
import numpy as np
import torch
import typeguard
from lsd.post.fragments import watershed_from_affinities as _watershed_lsd

from zetta_utils import builder, tensor_ops
from zetta_utils.tensor_ops.common import rearrange
from zetta_utils.tensor_ops.convert import to_np, to_torch
from zetta_utils.tensor_typing import TensorTypeVar

from .agglomeration import run_agglomeration_aff


@typeguard.typechecked
def run_watershed_abiss(
    affs: TensorTypeVar,
    aff_threshold_low: float = 0.01,
    aff_threshold_high: float = 0.99,
    size_threshold: int = 0,
) -> TensorTypeVar:
    """
    Produce watershed segments using the abiss package.

    :param affs: Affinity tensor in float32 with values [0.0, 1.0].
    :param aff_threshold_low, aff_threshold_high: Low and high watershed thresholds.
    :param size_threshold: If greater than 0, perform single-linkage merging as a
        subsequent step.
    :return: Segmented tensor.
    """
    affs_torch = torch.nn.functional.pad(
        to_torch(affs), (1, 1, 1, 1, 1, 1)
    )  # abiss requires 1px padding
    if affs_torch.shape[0] != 3:
        raise RuntimeError(
            f"Affinity channel should only have 3 components, receiving {affs.shape[0]}"
        )
    affs_torch = einops.rearrange(affs_torch, "C X Y Z -> X Y Z C")  # channel last
    affs_np = to_np(affs_torch)
    if affs_np.dtype == "uint8":
        # Assume affs are in range 0-255. Convert to 0.0-1.0 range.
        affs_np = affs_np.astype("float32") / 255
    assert affs_np.dtype == "float32"

    ret = abiss.watershed(
        affs=affs_np,
        aff_threshold_low=aff_threshold_low,
        aff_threshold_high=aff_threshold_high,
        size_threshold=size_threshold,
    )
    ret = ret[1:-1, 1:-1, 1:-1]
    ret = np.expand_dims(ret, axis=0)
    return tensor_ops.convert.astype(ret, affs)


@typeguard.typechecked
def run_watershed_lsd(
    affs: TensorTypeVar,
    fragments_in_xy: bool = False,
    min_seed_distance: int = 10,
    affs_in_xyz: bool = True,
) -> TensorTypeVar:
    """
    Produce watershed segments using the LSD package.

    :param affs: Affinity tensor in either float32 or uint8.
    :param fragments_in_xy: Produce supervoxels in xy
    :param min_seed_distance: Controls distance between seeds in voxels.
    :return: Segmented tensor.
    """
    """
    TODO:
    - add supervoxel filtering based on average aff value
    - add option to also perform agglomeration
    """
    affs_np = to_np(rearrange(affs, pattern="C X Y Z -> C Z Y X"))
    if affs_in_xyz:
        # aff needs to be zyx
        affs_np = np.flip(affs_np, 0)
    if affs_np.dtype == np.uint8:
        max_affinity_value = 255.0
        affs_np = affs_np.astype(np.float32)
    else:
        max_affinity_value = 1.0

    ret, _ = _watershed_lsd(
        affs=affs_np,
        max_affinity_value=max_affinity_value,
        fragments_in_xy=fragments_in_xy,
        min_seed_distance=min_seed_distance,
    )

    ret = einops.rearrange(ret, "Z Y X -> X Y Z")
    ret = np.expand_dims(ret, axis=0)
    return tensor_ops.convert.astype(ret, affs)


@builder.register("watershed_from_affinities")
@typeguard.typechecked
def watershed_from_affinities(
    affs: TensorTypeVar,  # in CXYZ
    method: Literal["abiss", "lsd"],
    agglomeration_threshold: float = 0.0,
    **kwargs,
) -> TensorTypeVar:
    """
    Produce supervoxels by running watershed on aff data. Optionally perform
    agglomeration and output segmentation.

    :param affs: Affinity tensor
    :param method: Method to use for watershed.
    :param agglomeration_threshold: If greater than 0.0, perform agglomeration
        as a subsequent step with this threshold.
    :return: Segmented tensor.
    """
    if method == "lsd":
        seg = run_watershed_lsd(affs, **kwargs)
    elif method == "abiss":
        seg = run_watershed_abiss(affs, **kwargs)

    if agglomeration_threshold > 0.0:
        seg, _ = run_agglomeration_aff(
            affs=affs,
            supervoxels=seg,
            threshold=agglomeration_threshold,
        )
    """
    TODO: write a wrapper for multi-chunk watershed that performs:
        - relabel supervoxels based on chunkid & chunk size
        - add supervoxel filtering based on mask
        - store a list of supervoxels within a chunk to a database
    """
    return seg

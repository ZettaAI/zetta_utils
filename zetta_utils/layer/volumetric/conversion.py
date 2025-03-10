from __future__ import annotations

from typing import Union

import numpy as np
import torch
from numpy import typing as npt

from zetta_utils import tensor_ops
from zetta_utils.geometry import Vec3D
from zetta_utils.geometry.bbox import BBox3D

from .index import VolumetricIndex

Slices3D = tuple[slice, slice, slice]
SliceUserVolumetricIndex = Union[
    tuple[Vec3D | None, BBox3D],
    tuple[slice, slice, slice],  # making the tuple explicit
    tuple[Vec3D | None, Slices3D],
    tuple[Vec3D | None, slice, slice, slice],
]

UnconvertedUserVolumetricIndex = Union[
    BBox3D,
    tuple[Vec3D | None, BBox3D],
    SliceUserVolumetricIndex,
]

UserVolumetricIndex = Union[
    UnconvertedUserVolumetricIndex,
    VolumetricIndex,
]


def get_bbox_from_user_vol_idx(
    idx_user: UnconvertedUserVolumetricIndex | VolumetricIndex, index_resolution: Vec3D | None
) -> BBox3D:
    """Extract a BBox3D from a user volumetric index.

    Args:
        idx_user: User volumetric index to extract bbox from
        index_resolution: Resolution to use for slices

    Returns:
        BBox3D extracted from the index

    Raises:
        ValueError: If index_resolution is None and resolution cannot be inferred
    """
    if isinstance(idx_user, VolumetricIndex):
        return idx_user.bbox

    result: BBox3D
    if isinstance(idx_user, BBox3D):
        result = idx_user
    elif len(idx_user) == 2 and isinstance(idx_user[1], BBox3D):
        result = idx_user[1]
    else:
        idx_slice: SliceUserVolumetricIndex = idx_user
        # MyPy is not aware of length checks, leading to the following type ignores
        if len(idx_slice) == 4:  # Tuple[Optional[Vec3D], slice, slice, sclie]
            specified_resolution: Vec3D | None = idx_slice[0]
            slices_user = idx_slice[1:]
        elif len(idx_slice) == 3:  # Tuple[slice, slice, sclie]
            specified_resolution = None
            slices_user = idx_slice
        else:  # Tuple[Optional[Vec3D], Tuple[slice, slice, sclie]]
            specified_resolution = idx_slice[0]
            slices_user = idx_slice[1]  # type: ignore
        if index_resolution is not None:
            slice_resolution = index_resolution
        elif specified_resolution is None:
            raise ValueError(
                f"Unable to convert {idx_user} to VolumetricResolution: cannot infer "
                "index resolution. Resolution not given as a part of index "
                "and `index_resolution` is None."
            )
        else:
            slice_resolution = specified_resolution

        result = BBox3D.from_slices(slices=slices_user, resolution=slice_resolution)

    return result


def get_desired_res_from_user_vol_idx(
    idx_user: UnconvertedUserVolumetricIndex | VolumetricIndex,
    default_desired_resolution: Vec3D | None,
) -> Vec3D:
    """Extract the desired resolution from a user volumetric index.

    Args:
        idx_user: User volumetric index to extract resolution from
        default_desired_resolution: Default resolution to use if not specified in the index

    Returns:
        Desired resolution extracted from the index

    Raises:
        ValueError: If default_desired_resolution is None and resolution cannot be inferred
    """
    if isinstance(idx_user, VolumetricIndex):
        return idx_user.resolution

    specified_resolution: Vec3D | None = None
    if isinstance(idx_user, tuple) and len(idx_user) != 3:  # not Tuple[slice, slice, sclie]
        specified_resolution = idx_user[0]

    if specified_resolution is not None:
        result = specified_resolution
    elif default_desired_resolution is not None:
        result = default_desired_resolution
    else:
        raise ValueError(
            f"Unable to convert {idx_user} to VolumetricResolution: cannot infer "
            "desired resolution. Resolution not given as a part of index "
            "and `default_desired_resolution` is None."
        )
    return result


def convert_idx(
    idx_user: UserVolumetricIndex,
    index_resolution: Vec3D | None,
    default_desired_resolution: Vec3D | None,
    allow_slice_rounding: bool,
) -> VolumetricIndex:
    """Convert a user volumetric index to a backend volumetric index.

    Args:
        idx_user: User volumetric index to convert
        index_resolution: Resolution of the index
        default_desired_resolution: Default desired resolution
        allow_slice_rounding: Whether to allow slice rounding

    Returns:
        Backend volumetric index
    """
    if isinstance(idx_user, VolumetricIndex):
        result = idx_user
    else:
        bbox = get_bbox_from_user_vol_idx(idx_user, index_resolution)
        desired_resolution = get_desired_res_from_user_vol_idx(
            idx_user, default_desired_resolution
        )

        result = VolumetricIndex(
            resolution=desired_resolution,
            bbox=bbox,
        )

    result.allow_slice_rounding = allow_slice_rounding
    return result


def convert_write(
    idx_user: UserVolumetricIndex,
    data_user: npt.NDArray | torch.Tensor | float | int | bool,
    index_resolution: Vec3D | None,
    default_desired_resolution: Vec3D | None,
    allow_slice_rounding: bool,
) -> tuple[VolumetricIndex, npt.NDArray]:
    """Convert a user volumetric index and data to backend format.

    Args:
        idx_user: User volumetric index to convert
        data_user: User data to convert
        index_resolution: Resolution of the index
        default_desired_resolution: Default desired resolution
        allow_slice_rounding: Whether to allow slice rounding

    Returns:
        Tuple of (backend volumetric index, backend data)
    """
    idx = convert_idx(
        idx_user,
        index_resolution,
        default_desired_resolution,
        allow_slice_rounding,
    )
    if isinstance(data_user, (float, int, bool)):
        dtype_mapping = {
            float: np.dtype("float32"),
            int: np.dtype("int32"),
            bool: np.dtype("int32"),
        }
        dtype = dtype_mapping[type(data_user)]
        data = np.array([data_user]).astype(dtype)
    else:
        data = tensor_ops.convert.to_np(data_user)

    return idx, data

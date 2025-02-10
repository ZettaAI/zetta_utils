from __future__ import annotations

from typing import Optional, Union

import attrs
import numpy as np
import torch
from numpy import typing as npt

from zetta_utils import tensor_ops
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.frontend_base import Frontend

from . import VolumetricIndex

Slices3D = tuple[slice, slice, slice]
SliceUserVolumetricIndex = Union[
    tuple[Optional[Vec3D], BBox3D],
    tuple[slice, slice, slice],  # making the tuple explicit
    tuple[Optional[Vec3D], Slices3D],
    tuple[Optional[Vec3D], slice, slice, slice],
]

UnconvertedUserVolumetricIndex = Union[
    BBox3D,
    tuple[Optional[Vec3D], BBox3D],
    SliceUserVolumetricIndex,
]

UserVolumetricIndex = Union[
    UnconvertedUserVolumetricIndex,
    VolumetricIndex,
]

UserVolumetricDataT = Union[npt.NDArray, torch.Tensor, float, int, bool]

@attrs.frozen
class VolumetricFrontend(Frontend[UserVolumetricIndex, VolumetricIndex, UserVolumetricDataT, npt.NDArray]):
    index_resolution: Vec3D | None = None
    default_desired_resolution: Vec3D | None = None
    allow_slice_rounding: bool = False

    def _get_bbox_from_user_vol_idx(self, idx_user: UnconvertedUserVolumetricIndex) -> BBox3D:
        result: BBox3D
        if isinstance(idx_user, BBox3D):
            result = idx_user
        elif len(idx_user) == 2 and isinstance(idx_user[1], BBox3D):
            result = idx_user[1]
        else:
            idx_slice: SliceUserVolumetricIndex = idx_user
            # MyPy is on very aware of length checks, leading to the following type ignores
            if len(idx_slice) == 4:  # Tuple[Optional[Vec3D], slice, slice, sclie]
                specified_resolution: Optional[Vec3D] = idx_slice[0]
                slices_user: Slices3D = idx_slice[1:]
            elif len(idx_slice) == 3:  # Tuple[slice, slice, sclie]
                specified_resolution = None
                slices_user = idx_slice
            else:  # Tuple[Optional[Vec3D], Tuple[slice, slice, sclie]]
                specified_resolution = idx_slice[0]
                slices_user = idx_slice[1]  # type: ignore
            if self.index_resolution is not None:
                slice_resolution = self.index_resolution
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

    def _get_desired_res_from_user_vol_idx(
        self, idx_user: UnconvertedUserVolumetricIndex
    ) -> Vec3D:
        specified_resolution: Optional[Vec3D] = None
        if isinstance(idx_user, tuple) and len(idx_user) != 3:  # not Tuple[slice, slice, sclie]
            specified_resolution = idx_user[0]

        if specified_resolution is not None:
            result = specified_resolution
        elif self.default_desired_resolution is not None:
            result = self.default_desired_resolution
        else:
            raise ValueError(
                f"Unable to convert {idx_user} to VolumetricResolution: cannot infer "
                "desired resolution. Resolution not given as a part of index "
                "and `default_desired_resolution` is None."
            )
        return result

    def convert_idx(self, idx_user: UserVolumetricIndex) -> VolumetricIndex:
        if isinstance(idx_user, VolumetricIndex):
            result = VolumetricIndex(
                resolution=idx_user.resolution,
                bbox=idx_user.bbox,
                allow_slice_rounding=self.allow_slice_rounding
            )
        else:
            bbox = self._get_bbox_from_user_vol_idx(idx_user)
            desired_resolution = self._get_desired_res_from_user_vol_idx(idx_user)

            result = VolumetricIndex(
                resolution=desired_resolution,
                bbox=bbox,
                allow_slice_rounding=self.allow_slice_rounding
            )

        return result

    def convert_write(
        self,
        idx_user: UserVolumetricIndex,
        data_user: UserVolumetricDataT 
    ) -> tuple[VolumetricIndex, npt.NDArray]:
        idx = self.convert_idx(idx_user)
        if isinstance(data_user, (float, int)):
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

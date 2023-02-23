from __future__ import annotations

from typing import Optional, Union

import attrs
import torch

from zetta_utils.geometry import BBox3D, Vec3D

from .. import DataProcessor, IndexProcessor, JointIndexDataProcessor, Layer
from . import VolumetricBackend, VolumetricIndex

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


VolumetricDataProcT = Union[
    DataProcessor[torch.Tensor], JointIndexDataProcessor[torch.Tensor, VolumetricIndex]
]


@attrs.frozen
class VolumetricLayer(Layer[VolumetricIndex, torch.Tensor]):
    backend: VolumetricBackend
    readonly: bool = False

    index_procs: tuple[IndexProcessor[VolumetricIndex], ...] = ()
    read_procs: tuple[VolumetricDataProcT, ...] = ()
    write_procs: tuple[VolumetricDataProcT, ...] = ()

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
                specified_resolution: Optional[Vec3D] = idx_slice[0]  # type: ignore
                slices_user: Slices3D = idx_slice[1:]  # type: ignore
            elif len(idx_slice) == 3:  # Tuple[slice, slice, sclie]
                specified_resolution = None
                slices_user = idx_slice  # type: ignore
            else:  # Tuple[Optional[Vec3D], Tuple[slice, slice, sclie]]
                specified_resolution = idx_slice[0]  # type: ignore
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
            specified_resolution = idx_user[0]  # type: ignore # mypy unaware of length

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

    def _convert_idx(self, idx_user: UserVolumetricIndex) -> VolumetricIndex:
        if isinstance(idx_user, VolumetricIndex):
            result = idx_user
        else:
            bbox = self._get_bbox_from_user_vol_idx(idx_user)
            desired_resolution = self._get_desired_res_from_user_vol_idx(idx_user)

            result = VolumetricIndex(
                resolution=desired_resolution,
                bbox=bbox,
            )

        result.allow_slice_rounding = self.allow_slice_rounding
        return result

    def _convert_write(
        self, idx_user: UserVolumetricIndex, data_user: Union[torch.Tensor, float, int, bool]
    ) -> tuple[VolumetricIndex, torch.Tensor]:
        idx = self._convert_idx(idx_user)
        if isinstance(data_user, (float, int)):
            dtype_mapping = {
                float: torch.float32,
                int: torch.int32,
                bool: torch.int32,
            }
            dtype = dtype_mapping[type(data_user)]
            data = torch.Tensor([data_user]).to(dtype)
        else:
            data = data_user

        return idx, data

    def __getitem__(self, idx: UserVolumetricIndex) -> torch.Tensor:
        idx_backend = self._convert_idx(idx)
        return self.read_with_procs(idx=idx_backend)

    def __setitem__(self, idx: UserVolumetricIndex, data: torch.Tensor | float | int | bool):
        idx_backend, data_backend = self._convert_write(idx, data)
        self.write_with_procs(idx=idx_backend, data=data_backend)
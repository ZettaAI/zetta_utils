# pylint: disable=missing-docstring,no-self-use,unused-argument
from __future__ import annotations

from typing import Optional, Tuple, Union, get_origin

import attrs
import torch

from zetta_utils import builder
from zetta_utils.bcube import BoundingCube
from zetta_utils.typing import Slices3D, Vec3D

from ..frontend_base import Frontend
from . import VolumetricIndex

SliceUserVolumetricIndex = Union[
    Tuple[Optional[Vec3D], BoundingCube],
    Tuple[slice, slice, slice],  # making the tuple explicit
    Tuple[Optional[Vec3D], Slices3D],
    Tuple[Optional[Vec3D], slice, slice, slice],
]

UnconvertedUserVolumetricIndex = Union[
    BoundingCube,
    Tuple[Optional[Vec3D], BoundingCube],
    SliceUserVolumetricIndex,
]

UserVolumetricIndex = Union[
    UnconvertedUserVolumetricIndex,
    VolumetricIndex,
]


@builder.register("VolumetricIndexConverter")
@attrs.mutable
class VolumetricFrontend(Frontend):
    index_resolution: Optional[Vec3D] = None
    default_desired_resolution: Optional[Vec3D] = None
    allow_slice_rounding: bool = False

    def _get_bcube_from_user_vol_idx(
        self, idx_user: UnconvertedUserVolumetricIndex
    ) -> BoundingCube:
        # static type system  generally confused here because of use of len() and get_origin.
        # it understands neither

        result: BoundingCube
        if isinstance(idx_user, get_origin(BoundingCube)):  # type: ignore
            result = idx_user  # type: ignore
        elif len(idx_user) == 2 and isinstance(  # type: ignore
            idx_user[1], get_origin(BoundingCube)  # type: ignore
        ):
            result = idx_user[1]  # type: ignore # mypy unaware of length
        else:
            idx_slice: SliceUserVolumetricIndex = idx_user  # type: ignore
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

            result = BoundingCube.from_slices(slices=slices_user, resolution=slice_resolution)

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
            bcube = self._get_bcube_from_user_vol_idx(idx_user)
            desired_resolution = self._get_desired_res_from_user_vol_idx(idx_user)

            result = VolumetricIndex(
                resolution=desired_resolution,
                bcube=bcube,
            )

        result.allow_slice_rounding = self.allow_slice_rounding
        return result

    def convert_read_idx(self, idx_user: UserVolumetricIndex) -> VolumetricIndex:
        return self._convert_idx(idx_user)

    def convert_read_data(self, idx_user: UserVolumetricIndex, data: torch.Tensor) -> torch.Tensor:
        return data

    def convert_write(
        self, idx_user: UserVolumetricIndex, data_user: Union[torch.Tensor, float, int]
    ) -> Tuple[VolumetricIndex, torch.Tensor]:
        idx = self._convert_idx(idx_user)
        if isinstance(data_user, (float, int)):
            data = torch.Tensor([data_user])
        else:
            data = data_user

        return idx, data

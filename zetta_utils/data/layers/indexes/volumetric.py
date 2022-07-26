# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Tuple, Literal, Optional, Union, Iterable, Callable
import attrs
from typeguard import typechecked

import zetta_utils as zu
from zetta_utils.typing import Vec3D, Slices3D
from zetta_utils.data.basic_ops import InterpolationMode
from zetta_utils.data.processors import Interpolate
from zetta_utils.data.layers.indexes.base import (
    Index,
    IndexConverter,
    IndexAdjuster,
    IndexAdjusterWithProcessors,
)

RawVolumetricIndex = Union[
    Slices3D,
    Tuple[Optional[Vec3D], slice, slice, slice],
]


@typechecked
@attrs.frozen
class VolumetricIndex(Index):  # pylint: disable=too-few-public-methods
    resolution: Vec3D
    slices: Slices3D


@typechecked
@attrs.mutable
class VolumetricIndexConverter(
    IndexConverter[VolumetricIndex]
):  # pylint: disable=too-few-public-methods
    index_resolution: Optional[Vec3D] = None
    default_desired_resolution: Optional[Vec3D] = None

    def raw_idx_to_idx(self, raw_idx: RawVolumetricIndex) -> VolumetricIndex:
        if len(raw_idx) == 3:  # Tuple[slice, slice, sclie], default index
            specified_resolution = None
            slices_raw = raw_idx  # type: Tuple[slice, slice, slice] # type: ignore
        else:
            assert len(raw_idx) == 4
            specified_resolution = raw_idx[0]  # type: Vec3D # type: ignore
            slices_raw = raw_idx[1:]  # type: ignore # Dosn't know the idx[1:] type

        if self.index_resolution is not None:
            slice_resolution = self.index_resolution
        elif specified_resolution is not None:
            slice_resolution = specified_resolution
        else:
            raise ValueError(
                "VolumetrixIndexConverter unable to infer index resolution: resolution not given "
                f"as a part of index {raw_idx} while `self.index_resolution` is None."
            )

        bcube = zu.bcube.BoundingCube.from_slices(slices=slices_raw, resolution=slice_resolution)

        if specified_resolution is not None:
            desired_resolution = specified_resolution
        elif self.default_desired_resolution is not None:
            desired_resolution = self.default_desired_resolution
        else:
            raise ValueError(
                "VolumetrixIndexConverter unable to infer desired resolution: resolution not "
                f"given as a part of index {raw_idx} while `self.default_desired_resolution` "
                "is None."
            )
        slices_final = bcube.to_slices(desired_resolution)

        result = VolumetricIndex(
            resolution=desired_resolution,
            slices=slices_final,
        )

        return result


@typechecked
@attrs.mutable
class AdjustDataResolution(
    IndexAdjusterWithProcessors[VolumetricIndex]
):  # pylint: disable=too-few-public-methods
    data_resolution: Vec3D
    interpolation_mode: InterpolationMode

    def __call__(
        self, idx: VolumetricIndex, mode: Literal["read", "write"]
    ) -> Tuple[VolumetricIndex, Iterable[Callable]]:
        processors = []

        if idx.resolution == self.data_resolution:
            idx_final = idx
        else:
            bcube = zu.bcube.BoundingCube.from_slices(idx.slices, idx.resolution)
            idx_final = VolumetricIndex(
                slices=bcube.to_slices(self.data_resolution),
                resolution=self.data_resolution,
            )

            if mode == "read":
                scale_factor = tuple(idx_final.resolution[i] / idx.resolution[i] for i in range(3))
            else:
                scale_factor = tuple(idx.resolution[i] / idx_final.resolution[i] for i in range(3))

            processors.append(
                Interpolate(scale_factor=scale_factor, interpolation_mode=self.interpolation_mode)
            )

        return idx_final, processors


@typechecked
@attrs.mutable
class TranslateVolumetricIndex(
    IndexAdjuster[VolumetricIndex]
):  # pylint: disable=too-few-public-methods
    offset: Vec3D
    offset_resolution: Vec3D

    def __call__(self, idx: VolumetricIndex) -> VolumetricIndex:
        bcube = zu.bcube.BoundingCube.from_slices(slices=idx.slices, resolution=idx.resolution)
        bcube_trans = bcube.translate(self.offset, self.offset_resolution)
        result = VolumetricIndex(
            slices=bcube_trans.to_slices(idx.resolution),
            resolution=idx.resolution,
        )
        return result

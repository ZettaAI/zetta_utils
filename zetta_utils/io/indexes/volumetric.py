# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Tuple, Literal, Optional, Union, Iterable, Callable
import attrs
from typeguard import typechecked

import zetta_utils as zu
from zetta_utils.typing import Vec3D, Slices3D
from zetta_utils.tensor.ops import InterpolationMode
from zetta_utils.tensor.processors import Interpolate
from zetta_utils.io.indexes.base import (
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

    @classmethod
    def convert(
        cls,
        idx_raw: RawVolumetricIndex,
        index_resolution: Optional[Vec3D] = None,
        default_desired_resolution: Optional[Vec3D] = None,
    ):
        if len(idx_raw) == 3:  # Tuple[slice, slice, sclie], default index
            specified_resolution = None  # type: Optional[Vec3D]
            slices_raw = idx_raw  # type: Tuple[slice, slice, slice] # type: ignore
        else:
            assert len(idx_raw) == 4
            specified_resolution = idx_raw[0]  # type: ignore
            slices_raw = idx_raw[1:]  # type: ignore

        if specified_resolution is not None:
            desired_resolution = specified_resolution
        elif default_desired_resolution is not None:
            specified_resolution = default_desired_resolution
            desired_resolution = specified_resolution
        else:
            raise ValueError(
                f"Unable to convert {idx_raw} to VolumetricResolution: cannot infer "
                "desired resolutionresolution. Resolution not given as a part of index "
                "and `default_desired_resolution` is None."
            )

        if index_resolution is not None:
            slice_resolution = index_resolution
        else:
            slice_resolution = specified_resolution

        bcube = zu.bbox.BoundingCube.from_slices(slices=slices_raw, resolution=slice_resolution)
        slices_final = bcube.to_slices(desired_resolution)

        result = VolumetricIndex(
            resolution=desired_resolution,
            slices=slices_final,
        )

        return result


@typechecked
@attrs.mutable
class VolumetricIndexConverter(
    IndexConverter[VolumetricIndex]
):  # pylint: disable=too-few-public-methods
    index_resolution: Optional[Vec3D] = None
    default_desired_resolution: Optional[Vec3D] = None

    def __call__(self, idx_raw: RawVolumetricIndex) -> VolumetricIndex:
        result = VolumetricIndex.convert(
            idx_raw=idx_raw,
            index_resolution=self.index_resolution,
            default_desired_resolution=self.default_desired_resolution,
        )
        return result


@typechecked
def translate_volumetric_index(idx: VolumetricIndex, offset: Vec3D, resolution: Vec3D):
    bcube = zu.bbox.BoundingCube.from_slices(slices=idx.slices, resolution=idx.resolution)
    bcube_trans = bcube.translate(offset, resolution)
    result = VolumetricIndex(
        slices=bcube_trans.to_slices(idx.resolution),
        resolution=idx.resolution,
    )
    return result


# TODO: it's possible to simplify indexer creation such that the below definition would be:
# TranslateVolumetricIndex = IndexAdjuster[VolumetricIndex].from_func(translate_volumetric_index)
# This can be done through dynamically creating a class. For now it's not done for the sake of
# codebase simplicity. Same can be done for converters and other processors in general.
@typechecked
@attrs.mutable
class TranslateVolumetricIndex(
    IndexAdjuster[VolumetricIndex]
):  # pylint: disable=too-few-public-methods
    offset: Vec3D
    resolution: Vec3D

    def __call__(self, idx: VolumetricIndex) -> VolumetricIndex:
        result = translate_volumetric_index(
            idx=idx, offset=self.offset, resolution=self.resolution
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
            bcube = zu.bbox.BoundingCube.from_slices(idx.slices, idx.resolution)
            idx_final = VolumetricIndex(
                slices=bcube.to_slices(self.data_resolution),
                resolution=self.data_resolution,
            )

            if mode == "read":
                scale_factor = tuple(idx_final.resolution[i] / idx.resolution[i] for i in range(3))
            else:
                scale_factor = tuple(idx.resolution[i] / idx_final.resolution[i] for i in range(3))

            processors.append(Interpolate(scale_factor=scale_factor, mode=self.interpolation_mode))

        return idx_final, processors

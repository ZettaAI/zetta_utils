# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Tuple, Literal, Optional, Union, Iterable, Callable
import attrs
from typeguard import typechecked

from zetta_utils import builder, tensor_ops
from zetta_utils.typing import Vec3D, Slices3D
from zetta_utils.partial import ComparablePartial
from zetta_utils.bcube import BoundingCube

from .. import (
    LayerIndex,
    IndexConverter,
    IndexAdjuster,
    IndexAdjusterWithProcessors,
)


@builder.register("VolumetricIndex")
@typechecked
@attrs.mutable
class VolumetricIndex(LayerIndex):  # pylint: disable=too-few-public-methods
    resolution: Vec3D
    slices: Slices3D

    @classmethod
    def default_convert(cls, idx_raw: RawVolumetricIndex) -> VolumetricIndex:  # pragma: no cover
        return VolumetricIndexConverter()(idx_raw)


RawVolumetricIndex = Union[
    VolumetricIndex,
    Slices3D,
    Tuple[Optional[Vec3D], slice, slice, slice],
]


@builder.register("VolumetricIndexConverter")
@typechecked
@attrs.mutable
class VolumetricIndexConverter(
    IndexConverter[RawVolumetricIndex, VolumetricIndex]
):  # pylint: disable=too-few-public-methods
    index_resolution: Optional[Vec3D] = None
    default_desired_resolution: Optional[Vec3D] = None
    allow_rounding: bool = False

    def __call__(self, idx_raw: RawVolumetricIndex) -> VolumetricIndex:
        if isinstance(idx_raw, VolumetricIndex):
            result = idx_raw
        else:
            if len(idx_raw) == 3:  # Tuple[slice, slice, sclie], default index
                specified_resolution = None  # type: Optional[Vec3D]
                slices_raw = idx_raw  # type: Tuple[slice, slice, slice] # type: ignore
            else:
                assert len(idx_raw) == 4
                specified_resolution = idx_raw[0]  # type: ignore
                slices_raw = idx_raw[1:]  # type: ignore

            if specified_resolution is not None:
                desired_resolution = specified_resolution
            elif self.default_desired_resolution is not None:
                specified_resolution = self.default_desired_resolution
                desired_resolution = specified_resolution
            else:
                raise ValueError(
                    f"Unable to convert {idx_raw} to VolumetricResolution: cannot infer "
                    "desired resolutionresolution. Resolution not given as a part of index "
                    "and `default_desired_resolution` is None."
                )

            if self.index_resolution is not None:
                slice_resolution = self.index_resolution
            else:
                slice_resolution = specified_resolution

            bcube = BoundingCube.from_slices(slices=slices_raw, resolution=slice_resolution)
            slices_final = bcube.to_slices(desired_resolution, allow_rounding=self.allow_rounding)

            result = VolumetricIndex(
                resolution=desired_resolution,
                slices=slices_final,
            )

        return result


@typechecked
def translate_volumetric_index(
    idx: VolumetricIndex, offset: Vec3D, resolution: Vec3D, allow_rounding: bool = False
):
    bcube = BoundingCube.from_slices(slices=idx.slices, resolution=idx.resolution)
    bcube_trans = bcube.translate(offset, resolution)
    result = VolumetricIndex(
        slices=bcube_trans.to_slices(idx.resolution, allow_rounding=allow_rounding),
        resolution=idx.resolution,
    )
    return result


# TODO: it's possible to simplify indexer creation such that the below definition would be:
# TranslateVolumetricIndex = IndexAdjuster[VolumetricIndex].from_func(translate_volumetric_index)
# This can be done through dynamically creating a class. For now it's not done for the sake of
# codebase simplicity. Same can be done for converters and other processors in general.
@builder.register("TranslateVolumetricIndex")
@typechecked
@attrs.mutable
class TranslateVolumetricIndex(
    IndexAdjuster[VolumetricIndex]
):  # pylint: disable=too-few-public-methods
    offset: Vec3D
    resolution: Vec3D
    allow_rounding: bool = False

    def __call__(self, idx: VolumetricIndex) -> VolumetricIndex:
        result = translate_volumetric_index(
            idx=idx,
            offset=self.offset,
            resolution=self.resolution,
            allow_rounding=self.allow_rounding,
        )
        return result


@builder.register("AdjustResolution")
@typechecked
@attrs.mutable
class AdjustDataResolution(
    IndexAdjusterWithProcessors[VolumetricIndex]
):  # pylint: disable=too-few-public-methods
    data_resolution: Vec3D
    interpolation_mode: tensor_ops.InterpolationMode
    allow_rounding: bool = False

    def __call__(
        self, idx: VolumetricIndex, mode: Literal["read", "write"]
    ) -> Tuple[VolumetricIndex, Iterable[Callable]]:
        procs = []

        if idx.resolution == self.data_resolution:
            idx_final = idx
        else:
            bcube = BoundingCube.from_slices(idx.slices, idx.resolution)
            idx_final = VolumetricIndex(
                slices=bcube.to_slices(
                    self.data_resolution,
                    allow_rounding=self.allow_rounding,
                ),
                resolution=self.data_resolution,
            )

            if mode == "read":
                scale_factor = tuple(idx_final.resolution[i] / idx.resolution[i] for i in range(3))
            else:
                scale_factor = tuple(idx.resolution[i] / idx_final.resolution[i] for i in range(3))

            procs.append(
                ComparablePartial(
                    tensor_ops.interpolate,
                    scale_factor=scale_factor,
                    mode=self.interpolation_mode,
                    allow_shape_rounding=self.allow_rounding,
                    unsqueeze_input_to=5,  # b + c + xyz
                )
            )

        return idx_final, procs

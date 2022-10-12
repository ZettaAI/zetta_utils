# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Tuple, Optional, Union, get_origin
import attrs
from typeguard import typechecked

from zetta_utils import builder, tensor_ops
from zetta_utils.typing import Vec3D, Slices3D
#from zetta_utils.partial import ComparablePartial
from zetta_utils.bcube import BoundingCube

from .. import LayerIndex


@builder.register("VolumetricIndex")
@typechecked
@attrs.mutable
class VolumetricIndex(LayerIndex):
    resolution: Vec3D
    bcube: BoundingCube

    @classmethod
    def default_convert(cls, idx_raw: RawVolumetricIndex) -> VolumetricIndex:  # pragma: no cover
        return VolumetricIndexConverter()(idx_raw)

SliceRawVolumetricIndex = Union[
    Tuple[Optional[Vec3D], BoundingCube],
    Tuple[slice, slice, slice], # making the tuple explicit
    Tuple[Optional[Vec3D], Slices3D],
    Tuple[Optional[Vec3D], slice, slice, slice],

]

ConvertibleRawVolumetricIndex = Union[
    BoundingCube,
    Tuple[Optional[Vec3D], BoundingCube],
    SliceRawVolumetricIndex,
]

RawVolumetricIndex = Union[
    ConvertibleRawVolumetricIndex,
    VolumetricIndex,
]



@builder.register("VolumetricIndexConverter")
@typechecked
@attrs.mutable
class VolumetricIndexConverter:  # pylint: disable=too-few-public-methods
    index_resolution: Optional[Vec3D] = None
    default_desired_resolution: Optional[Vec3D] = None

    def _get_bcube_from_raw_vol_idx(self, idx_raw: ConvertibleRawVolumetricIndex) -> BoundingCube:
        # mypy generally confused here because of use of len() and  get_origin.
        # it understands neither
        if isinstance(idx_raw, get_origin(BoundingCube)): # type: ignore
            result = idx_raw # type: BoundingCube # type: ignore
        elif len(idx_raw) == 2 and isinstance(idx_raw[1], get_origin(BoundingCube)): # type: ignore
            result = idx_raw[1] # type: ignore # mypy unaware of length
        else:
            idx_slice: SliceRawVolumetricIndex = idx_raw # type: ignore
            if len(idx_slice) == 4: # Tuple[Optional[Vec3D], slice, slice, sclie]
                specified_resolution: Optional[Vec3D] = idx_slice[0] # type: ignore
                slices_raw: Slices3D = idx_slice[1:] # type: ignore
            elif len(idx_slice) == 3:  # Tuple[slice, slice, sclie]
                specified_resolution = None
                slices_raw = idx_slice # type: ignore
            else: # Tuple[Optional[Vec3D], Tuple[slice, slice, sclie]]
                specified_resolution = idx_slice[0] # type: ignore
                slices_raw = idx_slice[1] # type: ignore

            if self.index_resolution is not None:
                slice_resolution = self.index_resolution
            elif specified_resolution is None:
                raise ValueError(
                    f"Unable to convert {idx_raw} to VolumetricResolution: cannot infer "
                    "index resolution. Resolution not given as a part of index "
                    "and `index_resolution` is None."
                )
            else:
                slice_resolution = specified_resolution

            result = BoundingCube.from_slices(slices=slices_raw, resolution=slice_resolution)
        return result

    def _get_desired_res_from_raw_vol_idx(self, idx_raw: ConvertibleRawVolumetricIndex) -> Vec3D:
        specified_resolution: Optional[Vec3D] = None
        if isinstance(idx_raw, tuple) and len(idx_raw) != 3: # not Tuple[slice, slice, sclie]
            specified_resolution = idx_raw[0] # type: ignore # mypy unaware of length

        if specified_resolution is not None:
            result = specified_resolution
        elif self.default_desired_resolution is not None:
            result = self.default_desired_resolution
        else:
            raise ValueError(
                f"Unable to convert {idx_raw} to VolumetricResolution: cannot infer "
                "desired resolution. Resolution not given as a part of index "
                "and `default_desired_resolution` is None."
            )
        return result

    def __call__(self, idx_raw: RawVolumetricIndex) -> VolumetricIndex:
        if isinstance(idx_raw, VolumetricIndex):
            result = idx_raw
        else:
            bcube = self._get_bcube_from_raw_vol_idx(idx_raw)
            desired_resolution = self._get_desired_res_from_raw_vol_idx(idx_raw)

            result = VolumetricIndex(
                resolution=desired_resolution,
                bcube=bcube,
            )

        return result


@typechecked
def translate_volumetric_index(
    idx: VolumetricIndex, offset: Vec3D, resolution: Vec3D
):
    bcube = idx.bcube.translate(offset, resolution)
    result = VolumetricIndex(
        bcube=bcube,
        resolution=idx.resolution,
    )
    return result


@builder.register("VolIdxTranslator")
@typechecked
@attrs.mutable
class VolIdxTranslator:
    offset: Vec3D
    resolution: Vec3D

    def __call__(self, idx: VolumetricIndex) -> VolumetricIndex:
        result = translate_volumetric_index(
            idx=idx,
            offset=self.offset,
            resolution=self.resolution,
        )
        return result


@builder.register("VolIdxResolutionAdjuster")
@typechecked
@attrs.mutable
class VolIdxResolutionAdjuster:
    data_resolution: Vec3D
    interpolation_mode: tensor_ops.InterpolationMode

    def __call__(
        self, idx: VolumetricIndex
    ) -> VolumetricIndex:
        result = VolumetricIndex(
            bcube=idx.bcube,
            resolution=self.data_resolution,
        )
        return result

'''
def tmp():
    bcube = BoundingCube.from_slices(idx.slices, idx.resolution)


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
'''

from typing import Literal
from typeguard import typechecked
import attrs
from zetta_utils import builder, tensor_ops
from zetta_utils.typing import Vec3D
from zetta_utils.tensor_typing import TensorTypeVar

from .. import DataWithIndexProcessor
from . import VolumetricIndex


@typechecked
def translate_volumetric_index(
    idx: VolumetricIndex, offset: Vec3D, resolution: Vec3D
):  # pragma: no cover # under 3 statements, no conditionals
    bcube = idx.bcube.translate(offset, resolution)
    result = VolumetricIndex(
        bcube=bcube,
        resolution=idx.resolution,
    )
    return result


@builder.register("VolIdxTranslator")
@typechecked
@attrs.mutable
class VolIdxTranslator:  # pragma: no cover # under 3 statements, no conditionals
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
class VolIdxResolutionAdjuster:  # pragma: no cover # under 3 statements, no conditionals
    resolution: Vec3D

    def __call__(self, idx: VolumetricIndex) -> VolumetricIndex:
        result = VolumetricIndex(
            bcube=idx.bcube,
            resolution=self.resolution,
        )
        return result


@builder.register("VolIdxResolutionAdjuster")
@typechecked
@attrs.mutable
class VolDataInterpolator(DataWithIndexProcessor):
    interpolation_mode: tensor_ops.InterpolationMode
    mode: Literal["read", "write"]
    allow_shape_rounding: bool = False

    def __call__(
        self,
        data: TensorTypeVar,
        idx: VolumetricIndex,
        idx_proced: VolumetricIndex,
    ) -> TensorTypeVar:
        if self.mode == "read":
            scale_factor = tuple(idx_proced.resolution[i] / idx.resolution[i] for i in range(3))
        else:
            assert self.mode == "write"
            scale_factor = tuple(idx.resolution[i] / idx_proced.resolution[i] for i in range(3))

        result = tensor_ops.interpolate(
            data=data,
            scale_factor=scale_factor,
            mode=self.interpolation_mode,
            allow_shape_rounding=self.allow_shape_rounding,
            unsqueeze_input_to=5,  # b + c + xyz
        )

        return result

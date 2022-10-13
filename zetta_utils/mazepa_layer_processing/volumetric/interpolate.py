from typing import Optional
import attrs
from typeguard import typechecked
from zetta_utils import builder, tensor_ops
from zetta_utils.typing import Vec3D
from zetta_utils.layer import Layer
from zetta_utils.layer.volumetric import VolumetricIndex, RawVolumetricIndex
from .. import LayerProcessor


@builder.register("InterpolateVolLayer")
@typechecked
@attrs.frozen()
class InterpolateVolLayer(LayerProcessor):
    resolution: Vec3D
    mode: tensor_ops.InterpolationMode
    mask_value_thr: float = 0
    # preserve_zeros: bool = False

    def __call__(
        self,
        src: Layer[RawVolumetricIndex, VolumetricIndex],
        idx: VolumetricIndex,
        dst: Optional[Layer[RawVolumetricIndex, VolumetricIndex]] = None,
    ):
        if dst is None:
            dst = src
        data_src = src[idx]
        scale_factor = tuple(idx.resolution[i] / self.resolution[i] for i in range(3))
        data_dst = tensor_ops.interpolate(
            data=data_src,
            scale_factor=scale_factor,
            mode=self.mode,
            unsqueeze_input_to=5, # Only 3D data is allowed here -- no 2D!
        )
        idx_dst = VolumetricIndex(
            bcube=idx.bcube,
            resolution=self.resolution,
        )
        dst[idx_dst] = data_dst

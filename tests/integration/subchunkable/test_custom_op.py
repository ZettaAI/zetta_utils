# This file triggers a RecursionError inside id_generation.py.

from typing import Sequence

import attrs

import zetta_utils
import zetta_utils.mazepa_layer_processing.common
from zetta_utils import builder, mazepa
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.tensor_ops import convert


@builder.register("CustomVolOp")
@mazepa.taskable_operation_cls
@attrs.frozen()
class CustomVolOp:  # implements VolumetricOpProtocol
    crop_pad: Sequence[int] = (0, 0, 0)
    threshold: float = 0.1
    connectivity: int = 6
    ids_per_chunk: int = 10000

    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:  # pylint: disable=no-self-use
        # For simplicity, just return the destination resolution
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> "CustomVolOp":
        return attrs.evolve(self, crop_pad=Vec3D[int](*self.crop_pad) + crop_pad)

    def __call__(
        self, idx: VolumetricIndex, dst: VolumetricLayer, src: VolumetricLayer, *args, **kwargs
    ):
        # If the following line is commented out, the problem does not occur:
        data_np = convert.to_np(src[idx][0])  # (use channel 0) # pylint: disable=unused-variable


# d = {
#     "@type": "build_subchunkable_apply_flow",
#     "bbox": {
#         "@type": "BBox3D.from_coords",
#         "start_coord": [13996, 10633, 3062],
#         "end_coord": [14124, 10761, 3102],
#         "resolution": [16, 16, 42],
#     },
#     "dst_resolution": [16, 16, 42],
#     "processing_chunk_sizes": [[128, 128, 40]],
#     "skip_intermediaries": True,
#     "expand_bbox_processing": True,
#     "op": {"@type": "CustomVolOp"},
#     "op_kwargs": {
#         "src": {
#             "@type": "build_cv_layer",
#             "path": "gs://dkronauer-ant-001-synapse/test/syndet20240802",
#         }
#     },
#     "dst": {
#         "@type": "build_cv_layer",
#         "path": "gs://dkronauer-ant-001-synapse/test/synseg20240802",
#         "info_reference_path": "gs://dkronauer-ant-001-synapse/test/syndet20240802",
#         "on_info_exists": "overwrite",
#         "info_field_overrides": {"num_channels": 1, "data_type": "int32"},
#         "info_add_scales": [[16, 16, 42]],
#         "info_add_scales_mode": "replace",
#     },
# }
# flow = zetta_utils.builder.build(d)

from typing import Any, Callable, Optional
import copy
import attrs
import mazepa

from zetta_utils import builder
from zetta_utils.typing import Vec3D
from zetta_utils.layer import Layer

from zetta_utils.layer.volumetric import VolumetricIndex

from .. import ChunkedApply
from . import ComputeFieldTaskFactory


@builder.register("ComputeFieldTaskFactory")
@attrs.mutable
class _ComputeFieldTaskFactory:
    compute_field_method: Callable # TODO: type me
    tgt_offset: Vec3D

    def __call__(
        self,
        idx: VolumetricIndex,
        src: Layer[Any, VolumetricIndex],
        dst: Layer[Any, VolumetricIndex],
        tgt: Optional[Layer[Any, VolumetricIndex]] = None,
    ):
        src_data = src[idx]
        tgt_idx = copy.deepcopy(idx)
        tgt_idx.bcube.translate(
            offset=self.tgt_offset,
            resolution=tgt_idx.resolution,
        )
        if tgt is not None:
            tgt_data = tgt[idx]
        else:
            tgt_data = src[idx]

        result = self.compute_field_method(src_data, tgt_data)
        dst[idx] = result

# Using decorator breaks mypy
ComputeFieldTaskFactory = mazepa.task_factory_cls(_ComputeFieldTaskFactory)


@mazepa.flow_type
@typechecked
def compute_z_neighbor_fields(
    src: Layer[Any, VolumetricIndex],
    dst_dir: str,
    compute_field_method: Callable,
    idx: VolumetricIndex,
    chunker: IndexChunker[VolumetricIndex],
    dst_layer_builder: Callable[..., Layer[Any, VolumetricIndex]],
    dst_layer_prefix: str = "neighbor_field_z",
    farthest_neighbor: int = 3,
    direction: Literal["backward", "forward"] = "backward",
):
    if direction == "backward":
        z_offsets = range(-1, -farthest_neighbor - 1, -1)
    else:
        z_offsets = range(1, farthest_neighbor + 1)

    for z_offset in z_offsets:
        dst_path = os.path.join(dst_dir, f"{dst_layer_prefix}_{z_offset}")
        dst = dst_layer_builder(path=dst_path)

        comp_fact = ComputeFieldTaskFactory(
            compute_field_method=compute_field_method,
            tgt_offset=(0, 0, z_offset),
        )

        # TODO: chunked apply is completely untyped here. How to type it to check runtime args?
        comp_flow = ChunkedApply(
            chunker=chunker,
            task_factory=comp_fact,
        )(idx=idx, src=src, dst=dst)

        yield comp_flow

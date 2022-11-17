import copy
import os
from typing import Any, Callable, Literal

import torch
from typeguard import typechecked

from zetta_utils import builder, mazepa
from zetta_utils.layer import IndexChunker, Layer
from zetta_utils.layer.volumetric import VolIdxTranslator, VolumetricIndex

from .. import ChunkedApplyFlowType


@builder.register("compute_z_neighbor_fields")
@mazepa.flow_type
@typechecked
def compute_z_neighbor_fields(
    src: Layer[Any, VolumetricIndex, torch.Tensor],
    dst_dir: str,
    cf_task_factory: mazepa.TaskFactory,  # TODO: type me
    idx: VolumetricIndex,
    chunker: IndexChunker[VolumetricIndex],
    dst_layer_builder: Callable[..., Layer[Any, VolumetricIndex, torch.Tensor]],
    dst_layer_prefix: str = "neighbor_field_z",
    farthest_neighbor: int = 1,
    direction: Literal["backward", "forward"] = "backward",
):
    """
    :param cf_task_factory: Compute field task factory. Must take in the following
        kwargs: {'idx': VolumetricIndex, 'src': Layer, 'tgt': Layer, 'dst': Layer}
    """
    if direction == "backward":
        z_offsets = range(-1, -farthest_neighbor - 1, -1)
    else:
        z_offsets = range(3, farthest_neighbor + 2)

    for z_offset in z_offsets:
        dst_path = os.path.join(dst_dir, f"{dst_layer_prefix}_{z_offset}")
        dst = dst_layer_builder(path=dst_path)

        tgt = copy.deepcopy(src)
        tgt.index_adjs.insert(
            0, VolIdxTranslator(offset=(0, 0, z_offset), resolution=idx.resolution)
        )

        cf_flow = ChunkedApplyFlowType(
            task_factory=cf_task_factory,
            chunker=chunker,
        )(idx=idx, dst=dst, src=src, tgt=tgt)

        yield cf_flow

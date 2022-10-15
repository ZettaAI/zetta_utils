from typing import Any, Dict, Callable
import os
import mazepa
from typeguard import typechecked

from zetta_utils.layer import Layer, IndexChunker
from zetta_utils.layer.volumetric import VolumetricIndex
from .. import chunked_job
from . import ComputeFieldProcessor


@mazepa.job
@typechecked
def aced_compute_neighbor_fields(
    layers: Dict[str, Layer[Any, VolumetricIndex]],
    dst_dir: str,
    compute_field_method: Callable,
    idx: VolumetricIndex,
    chunker: IndexChunker[VolumetricIndex],
    dst_layer_builder: Callable[..., Layer[Any, VolumetricIndex]],
    dst_layer_prefix: str = "neighbor_field",
    farthest_neighbor: int = 3,
):
    for z_offset in range(-1, -farthest_neighbor - 1, -1):
        dst_path = os.path.join(dst_dir, f"{dst_layer_prefix}_{z_offset}")
        layers["dst"] = dst_layer_builder(path=dst_path)
        comp_proc = ComputeFieldProcessor(
            compute_field_method=compute_field_method,
            tgt_offset=(0, 0, z_offset),
        )
        comp_job = chunked_job(layers=layers, idx=idx, processor=comp_proc, chunker=chunker)
        yield comp_job

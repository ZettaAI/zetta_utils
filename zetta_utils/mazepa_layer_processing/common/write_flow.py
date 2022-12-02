from typing import Any, TypeVar

from zetta_utils import builder, mazepa
from zetta_utils.bcube import BoundingCube
from zetta_utils.layer import IndexChunker, Layer, LayerIndex
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricIndexChunker
from zetta_utils.typing import IntVec3D, Vec3D

from . import build_chunked_callable_flow_schema

IndexT = TypeVar("IndexT", bound=LayerIndex)


def _write_callable(src):
    return src


@builder.register("generic_write_flow")
def generic_write_flow(
    chunker: IndexChunker[IndexT],
    idx: IndexT,
    dst: Layer[Any, IndexT, Any],
    src: Layer[Any, IndexT, Any],
) -> mazepa.Flow:
    flow_schema = build_chunked_callable_flow_schema(
        fn=_write_callable,
        chunker=chunker,
    )
    result = flow_schema(idx=idx, dst=dst, src=src)
    return result


@builder.register("build_write_flow")
def build_write_flow(
    chunk_size: IntVec3D,
    bcube: BoundingCube,
    dst_resolution: Vec3D,
    dst: Layer[Any, VolumetricIndex, Any],
    src: Layer[Any, VolumetricIndex, Any],
) -> mazepa.Flow:
    result = generic_write_flow(
        chunker=VolumetricIndexChunker(chunk_size=chunk_size),
        idx=VolumetricIndex(bcube=bcube, resolution=dst_resolution),
        src=src,
        dst=dst,
    )
    return result

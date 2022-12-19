from os import path
from typing import Generic, Iterable, List, Literal, Optional, Tuple, TypeVar

import attrs
from typing_extensions import ParamSpec

from zetta_utils import builder, log, mazepa, tensor_ops
from zetta_utils.bcube import BoundingCube
from zetta_utils.layer.volumetric import (
    VolumetricDataBlendingWeighter,
    VolumetricIndex,
    VolumetricIndexChunker,
    VolumetricLayer,
)
from zetta_utils.typing import IntVec3D, Vec3D

from ..operation_protocols import BlendableOpProtocol

logger = log.get_logger("zetta_utils")

IndexT = TypeVar("IndexT")
P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)


@mazepa.taskable_operation
def reduce_by_sum(
    src_layers: List[VolumetricLayer], idx: VolumetricIndex, dst: VolumetricLayer
) -> None:
    assert len(src_layers) > 0
    res = src_layers[0][idx]
    for layer in src_layers[1:]:
        res = tensor_ops.common.add(res, layer[idx])
    dst[idx] = res


@builder.register(
    "BlendableApplyFlowSchema",
    cast_to_vec3d=["dst_resolution"],
    cast_to_intvec3d=["chunk_size", "blend_pad"],
)
@mazepa.flow_schema_cls
@attrs.mutable
class BlendableApplyFlowSchema(Generic[P, R_co]):
    operation: BlendableOpProtocol[P, R_co]
    chunk_size: IntVec3D
    dst_resolution: Vec3D
    blend_pad: Optional[IntVec3D] = None
    blend_mode: Literal["linear", "quadratic"] = "linear"
    temp_layers_dir: Optional[str] = None
    chunker: VolumetricIndexChunker = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.chunker = VolumetricIndexChunker(
            chunk_size=self.chunk_size, resolution=self.dst_resolution
        )

    def make_tasks_without_blending(
        self, idx_chunks: Iterable[VolumetricIndex], dst: VolumetricLayer, **kwargs: P.kwargs
    ) -> List[mazepa.tasks.Task[R_co]]:
        tasks = [
            self.operation.make_task(
                idx=idx_chunk,
                dst=dst,
                **kwargs,
            )
            for idx_chunk in idx_chunks
        ]

        return tasks

    def make_tasks_with_blending(
        self,
        idx: VolumetricIndex,
        idx_chunks: Iterable[VolumetricIndex],
        dst: VolumetricLayer,
        **kwargs: P.kwargs,
    ) -> Tuple[List[mazepa.tasks.Task[R_co]], List[List[VolumetricLayer]]]:
        """Makes tasks that will generate tasks that can be reduced to the final output,
        alongside the list of the temporary layers for each chunk in idx_chunks that need to be
        reduced for the final output in that chunk."""
        if self.temp_layers_dir is None:
            raise ValueError("`temp_layers_dir` must be specified when using blending")
        if self.blend_pad is None:
            raise ValueError("`blend_pad` must be specified when using blending")

        blending_weighter = VolumetricDataBlendingWeighter(self.blend_pad, self.blend_mode)

        tasks = []
        idx_chunks_temps: List[List[VolumetricLayer]] = [[] for _ in idx_chunks]
        for chunker, chunker_idx in self.chunker.split_into_nonoverlapping_chunkers(
            self.blend_pad
        ):
            dst_temp = dst.clone(
                name=path.join(
                    self.temp_layers_dir,
                    f"_{self.operation.__class__.__name__}_temp_{chunker_idx}",
                )
            )
            dst_temp.write_preprocs.append(blending_weighter)
            dst_temp.backend.enforce_chunk_aligned_writes = False

            # padding for the edges of the idx so they don't get affected by blending,
            # as well as bbox rounding down
            chunker_idx_chunks = chunker(
                idx.translate_start(-2 * self.blend_pad).translate_stop(
                    self.chunk_size + 2 * self.blend_pad
                )
            )
            for chunker_idx_chunk in chunker_idx_chunks:
                tasks.append(self.operation.make_task(chunker_idx_chunk, dst_temp, **kwargs))
                for i, idx_chunk in enumerate(idx_chunks):
                    if chunker_idx_chunk.intersects(idx_chunk):
                        idx_chunks_temps[i].append(dst_temp)

        return tasks, idx_chunks_temps

    def flow(
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> mazepa.FlowFnReturnType:
        # can't figure out how to force mypy to check this
        assert len(args) == 0
        logger.info(f"Breaking {idx} into chunks with {self.chunker}.")
        idx_chunks = self.chunker(idx)
        # case without blending
        if self.blend_pad is None:
            tasks = self.make_tasks_without_blending(idx_chunks, dst, **kwargs)
            logger.info(
                f"Submitting {len(tasks)} processing tasks from operation {self.operation}."
            )
            yield tasks
        # case with blending
        else:
            tasks, idx_chunks_temps = self.make_tasks_with_blending(idx, idx_chunks, dst, **kwargs)
            logger.info(
                "Writing to temporary destinations:\n"
                + f"Submitting {len(tasks)} processing tasks from operation {self.operation}."
            )
            yield tasks
            yield mazepa.Dependency()
            tasks_reduce = [
                reduce_by_sum.make_task(src_layers=idx_chunk_temps, idx=idx_chunk, dst=dst)
                for idx_chunk_temps, idx_chunk in zip(idx_chunks_temps, idx_chunks)
            ]
            logger.info(
                "Collating temporary destination CloudVolumes into the final destination:"
                + f"Submitting {len(tasks_reduce)} tasks."
            )
            yield tasks_reduce


@builder.register(
    "build_blendable_apply_flow",
    cast_to_vec3d=["dst_resolution"],
    cast_to_intvec3d=["blend_pad", "chunk_size"],
)
def build_blendable_apply_flow(  # pylint: disable=keyword-arg-before-vararg
    operation: BlendableOpProtocol[P, R_co],
    bcube: BoundingCube,
    dst_resolution: Vec3D,
    chunk_size: IntVec3D,
    blend_pad: Optional[IntVec3D] = None,
    blend_mode: Literal["linear", "quadratic"] = "linear",
    temp_layers_dir: Optional[str] = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> mazepa.Flow:
    idx = VolumetricIndex(resolution=dst_resolution, bcube=bcube)
    flow_schema: BlendableApplyFlowSchema[P, R_co] = BlendableApplyFlowSchema(
        operation=operation,
        chunk_size=chunk_size,
        dst_resolution=dst_resolution,
        blend_pad=blend_pad,
        blend_mode=blend_mode,
        temp_layers_dir=temp_layers_dir,
    )
    flow = flow_schema(idx, *args, **kwargs)

    return flow

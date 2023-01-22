from copy import deepcopy
from os import path
from typing import Generic, Iterable, List, Literal, Optional, Tuple, TypeVar

import attrs
import torch
from typing_extensions import ParamSpec

from zetta_utils import builder, log, mazepa, tensor_ops
from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricIndexChunker,
    VolumetricLayer,
)

from ..operation_protocols import BlendableOpProtocol

logger = log.get_logger("zetta_utils")

IndexT = TypeVar("IndexT")
P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)


def get_intersection_and_subindex(small: VolumetricIndex, large: VolumetricIndex):
    """
    Given a large VolumetricIndex and a small VolumetricIndex, returns the intersection
    VolumetricIndex of the two as well as the slices for that intersection within the
    large VolumetricIndex.
    """
    intersection = large.intersection(small)
    subindex = [
        slice(start - offset, stop - offset)
        for start, stop, offset in zip(intersection.start, intersection.stop, large.start)
    ]
    return intersection, subindex


@mazepa.taskable_operation
def reduce_by_weighted_sum(
    src_idxs: List[VolumetricIndex],
    src_layers: List[VolumetricLayer],
    src_layers_weights: List[VolumetricLayer],
    idx: VolumetricIndex,
    dst: VolumetricLayer,
    use_weights: bool,
) -> None:
    res = torch.zeros_like(dst[idx])
    assert len(src_layers) > 0
    if use_weights:
        for src_idx, layer, weight in zip(src_idxs, src_layers, src_layers_weights):
            intscn, subidx = get_intersection_and_subindex(src_idx, idx)
            subidx.insert(0, slice(0, res.shape[0]))
            res[subidx] = tensor_ops.common.add(res[subidx], layer[intscn] * weight[intscn])
    else:
        for src_idx, layer in zip(src_idxs, src_layers):
            intscn, subidx = get_intersection_and_subindex(src_idx, idx)
            subidx.insert(0, slice(0, res.shape[0]))
            res[subidx] = tensor_ops.common.add(res[subidx], layer[intscn])
    dst[idx] = res


@mazepa.taskable_operation
def write_blending_weights(  # pylint:disable=too-many-branches
    dst: VolumetricLayer,
    idx_subchunk: VolumetricIndex,
    idx_chunk: VolumetricIndex,
    blend_pad: IntVec3D,
    blend_mode: Literal["linear", "quadratic"] = "linear",
) -> None:
    """
    Writes the correct blending weights for an `idx_subchunk` inside a `idx_chunk` at `dst`,
    suppressing the weights for the dimension(s) where `idx_subchunk` is aligned to the edge
    of `idx_chunk`.
    """
    if blend_pad == IntVec3D(0, 0, 0):
        raise ValueError("`blend_pad` must be nonzero to need blending weights")
    if not idx_subchunk.intersects(idx_chunk):
        raise ValueError(
            "`idx_chunk` must intersect `idx_subchunk`;"
            " `idx_chunk`: {idx_chunk}, `idx_subchunk`: {idx_subchunk}"
        )
    # This is necessary because we do not know a priori how many channels the data has.
    data = dst[idx_subchunk]
    try:
        for s, p in zip(IntVec3D(*data.shape[-3:]), blend_pad):  # pylint: disable=invalid-name
            assert s >= 2 * p
    except Exception as e:
        raise ValueError(
            f"received {tuple(data.shape[-3:])} data, expected at least {2*blend_pad}"
        ) from e
    mask = torch.ones_like(data, dtype=torch.float)

    # TODO: Tensor.flip is slow because it reallocates
    if blend_mode == "linear":
        """
        This is a much faster way to do
        if idx_subchunk.start[1] != idx_chunk.start[1]
            for y in range(2 * y_pad):
                weight = y / (2 * y_pad)
                mask[:, :, y, :] *= weight
                while
        if idx_subchunk.stop[1] != idx_chunk.stop[1]
            for y in range(2 * y_pad):
                weight = y / (2 * y_pad)
                mask[:, :, -y, :] *= weight
        and so forth.
        """
        for dim in range(3):
            pad_in_dim = blend_pad[dim]
            if pad_in_dim == 0:
                continue
            weights = torch.zeros(
                2 * pad_in_dim, dtype=mask.dtype, layout=mask.layout, device=mask.device
            )
            for i in range(2 * pad_in_dim):
                weights[i] = i / (2 * pad_in_dim)
            weights = weights.reshape(
                tuple(2 * pad_in_dim if i == dim + 1 else 1 for i in range(4))
            )
            if idx_subchunk.start[dim] != idx_chunk.start[dim]:
                mask[
                    [
                        slice(0, 2 * pad_in_dim) if i == dim + 1 else slice(0, mask.shape[i])
                        for i in range(4)
                    ]
                ] *= weights
            if idx_subchunk.stop[dim] != idx_chunk.stop[dim]:
                mask[
                    [
                        slice(-2 * pad_in_dim, mask.shape[i])
                        if i == dim + 1
                        else slice(0, mask.shape[i])
                        for i in range(4)
                    ]
                ] *= weights.flip(dim + 1)
    elif blend_mode == "quadratic":
        """
        This is a much faster way to do
        if idx_subchunk.start[0] != idx_chunk.start[0]
            for x in range(x_pad):
                weight = ((x / x_pad) ** 2) / 2
                mask[:, x, :, :] *= weight
                mask[:, 2 * x_pad - x, :, :] *= 1 - weight
        if idx_subchunk.start[0] != idx_chunk.start[0]
            for x in range(x_pad):
                weight = ((x / x_pad) ** 2) / 2
                mask[:, -x, :, :] *= weight
                mask[:, -(2 * x_pad - x), :, :] *= 1 - weight
        and so forth.
        """
        for dim in range(3):
            pad_in_dim = blend_pad[dim]
            if pad_in_dim == 0:
                continue
            weights = torch.zeros(
                pad_in_dim, dtype=mask.dtype, layout=mask.layout, device=mask.device
            )
            for i in range(pad_in_dim):
                weights[i] = ((i / pad_in_dim) ** 2) / 2
            weights = weights.reshape(tuple(pad_in_dim if i == dim + 1 else 1 for i in range(4)))
            if idx_subchunk.start[dim] != idx_chunk.start[dim]:
                mask[
                    [
                        slice(0, pad_in_dim) if i == dim + 1 else slice(0, mask.shape[i])
                        for i in range(4)
                    ]
                ] *= weights
                mask[
                    [
                        slice(pad_in_dim, 2 * pad_in_dim)
                        if i == dim + 1
                        else slice(0, mask.shape[i])
                        for i in range(4)
                    ]
                ] *= 1 - weights.flip(dim + 1)
            if idx_subchunk.stop[dim] != idx_chunk.stop[dim]:
                mask[
                    [
                        slice(-2 * pad_in_dim, -pad_in_dim)
                        if i == dim + 1
                        else slice(0, mask.shape[i])
                        for i in range(4)
                    ]
                ] *= (1 - weights)
                mask[
                    [
                        slice(-pad_in_dim, mask.shape[i])
                        if i == dim + 1
                        else slice(0, mask.shape[i])
                        for i in range(4)
                    ]
                ] *= weights.flip(dim + 1)
    dst[idx_subchunk] = mask


@builder.register("BlendableApplyFlowSchema")
@mazepa.flow_schema_cls
@attrs.mutable
class BlendableApplyFlowSchema(Generic[P, R_co]):
    operation: BlendableOpProtocol[P, R_co]
    processing_chunk_size: IntVec3D
    dst_resolution: Vec3D
    max_reduction_chunk_size: Optional[IntVec3D] = None
    crop_pad: Optional[IntVec3D] = None
    blend_pad: Optional[IntVec3D] = None
    blend_mode: Literal["linear", "quadratic"] = "linear"
    temp_layers_dir: Optional[str] = None
    expand: Optional[bool] = False
    use_checkerboarding: bool = attrs.field(init=False)
    use_checkerboarding_weights: bool = attrs.field(init=False)
    processing_chunker: VolumetricIndexChunker = attrs.field(init=False)

    def _get_backend_chunk_size_to_use(self, dst) -> IntVec3D:
        backend_chunk_size = deepcopy(self.processing_chunk_size)
        dst_backend_chunk_size = dst.backend.get_chunk_size(self.dst_resolution)
        for i in range(3):
            if backend_chunk_size[i] == 1:
                continue
            if self.processing_chunk_size[i] % 2 != 0:
                raise ValueError(
                    "`processing_chunk_size` must be divisible by 2 at least once in"
                    " dimensions that are not 1;"
                    " received {self.processing_chunk_size[i]}"
                )
            backend_chunk_size[i] //= 2
            while backend_chunk_size[i] > dst_backend_chunk_size[i]:
                if backend_chunk_size[i] % 2 != 0:
                    raise ValueError(
                        "`processing_chunk_size` must be divisible by 2 continuously until"
                        " it is smaller than the `dst` backend's chunk size, or be 1, in"
                        f" dims that are blended; received {self.processing_chunk_size[i]}"
                    )
                backend_chunk_size[i] //= 2

        return backend_chunk_size

    def __attrs_post_init__(self):
        if self.crop_pad is None:
            self.crop_pad = IntVec3D(0, 0, 0)
        if self.blend_pad is None:
            self.blend_pad = IntVec3D(0, 0, 0)
        if self.crop_pad != IntVec3D(0, 0, 0) or self.blend_pad != IntVec3D(0, 0, 0):
            self.use_checkerboarding = True
        else:
            # even if crop_pad and blend_pad are both zero,
            # use checkerboarding if max_reduction_chunk is set
            if self.max_reduction_chunk_size is not None:
                self.use_checkerboarding = True
                logger.info(
                    "Using checkerboarding even though `crop_pad` and `blend_pad` are zero"
                    " since `max_reduction_chunk_size` is nonzero;"
                    " received {self.max_reduction_chunk_size}"
                )
            else:
                self.use_checkerboarding = False
                self.use_checkerboarding_weights = False
        if self.use_checkerboarding:
            if not self.blend_pad <= self.processing_chunk_size // 2:
                raise ValueError(
                    f" `blend_pad` must be less than or equal to"
                    f" half of `processing_chunk_size`; received {self.crop_pad + self.blend_pad}",
                    f" which is larger than {self.processing_size // 2}",
                )
            if self.temp_layers_dir is None:
                raise ValueError("`temp_layers_dir` must be specified when using blending or crop")
            if self.blend_pad != IntVec3D(0, 0, 0):
                self.use_checkerboarding_weights = True
            else:
                self.use_checkerboarding_weights = False

        self.processing_chunker = VolumetricIndexChunker(
            chunk_size=self.processing_chunk_size, resolution=self.dst_resolution
        )
        if self.max_reduction_chunk_size is None:
            self.max_reduction_chunk_size = self.processing_chunk_size

    def make_tasks_without_checkerboarding(
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

    def make_tasks_with_checkerboarding(  # pylint: disable=too-many-locals
        self,
        idx: VolumetricIndex,
        red_chunks: Iterable[VolumetricIndex],
        dst: VolumetricLayer,
        **kwargs: P.kwargs,
    ) -> Tuple[
        List[mazepa.tasks.Task[R_co]],
        List[mazepa.tasks.Task[None]],
        List[List[VolumetricIndex]],
        List[List[VolumetricLayer]],
        List[List[VolumetricLayer]],
    ]:
        assert self.temp_layers_dir is not None
        assert self.blend_pad is not None
        """
        Makes tasks that can be reduced to the final output, tasks that write weights for the
        reduction, and for each reduction chunk, the list of task indices, the list of temporary
        output layers, and the list of temporary layers with weights for all tasks that intersect
        the reduction chunk that need to be reduced for the output.
        """
        tasks: List[mazepa.tasks.Task[R_co]] = []
        tasks_weights: List[mazepa.tasks.Task[None]] = []
        red_chunks_task_idxs: List[List[VolumetricIndex]] = [[] for _ in red_chunks]
        red_chunks_temps: List[List[VolumetricLayer]] = [[] for _ in red_chunks]
        red_chunks_temps_weights: List[List[VolumetricLayer]] = [[] for _ in red_chunks]
        for chunker, chunker_idx in self.processing_chunker.split_into_nonoverlapping_chunkers(
            self.blend_pad
        ):

            dst_temp = dst.clone(
                name=path.join(
                    self.temp_layers_dir,
                    f"_{self.operation.__class__.__name__}_temp_{idx.pformat()}_{chunker_idx}",
                )
            )
            """
            Prepare the temporary destination by allowing non-aligned writes, aligning the voxel
            offset of the temporary destination with where the given idx starts, and setting the
            backend chunk size to half of the processing chunk size.
            """
            dst_temp.backend.enforce_chunk_aligned_writes = False
            dst_temp.backend.set_voxel_offset(idx.start, self.dst_resolution)
            dst_temp.backend.set_chunk_size(
                self._get_backend_chunk_size_to_use(dst), self.dst_resolution
            )
            # assert that the idx passed in is in fact exactly divisible by the chunk size
            red_chunk_aligned = idx.snapped(
                grid_offset=idx.start, grid_size=self.processing_chunk_size, mode="expand"
            )
            if red_chunk_aligned != idx:
                raise ValueError(
                    f"received (crop padded) idx {idx} is not evenly divisible by"
                    f" {self.processing_chunk_size}"
                )
            if self.use_checkerboarding_weights:
                dst_temp_weights = dst_temp.clone(
                    name=path.join(
                        self.temp_layers_dir,
                        f"_{self.operation.__class__.__name__}_temp"
                        f"_{idx.pformat()}_{chunker_idx}_weights",
                    )
                )
            else:
                dst_temp_weights = None
            # expand to allow for blend_pad around the edges
            idx_expanded = idx.padded(self.blend_pad)
            task_idxs = chunker(idx_expanded)

            for task_idx in task_idxs:
                tasks.append(self.operation.make_task(task_idx, dst_temp, **kwargs))
                if self.use_checkerboarding_weights:
                    # needed for mypy
                    assert dst_temp_weights is not None
                    tasks_weights.append(
                        write_blending_weights.make_task(
                            dst=dst_temp_weights,
                            idx_subchunk=task_idx,
                            idx_chunk=idx_expanded,
                            blend_pad=self.blend_pad,
                            blend_mode=self.blend_mode,
                        )
                    )
                for i, red_chunk in enumerate(red_chunks):
                    if task_idx.intersects(red_chunk):
                        red_chunks_task_idxs[i].append(task_idx)
                        red_chunks_temps[i].append(dst_temp)
                        if self.use_checkerboarding_weights:
                            # needed for mypy
                            assert dst_temp_weights is not None
                            red_chunks_temps_weights[i].append(dst_temp_weights)

        return (
            tasks,
            tasks_weights,
            red_chunks_task_idxs,
            red_chunks_temps,
            red_chunks_temps_weights,
        )

    def flow(
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> mazepa.FlowFnReturnType:

        assert self.crop_pad is not None
        assert self.blend_pad is not None
        dst.backend.assert_idx_is_chunk_aligned(idx)

        # can't figure out how to force mypy to check this
        assert len(args) == 0
        logger.info(f"Breaking {idx} into chunks with {self.processing_chunker}.")

        # case without checkerboarding
        if not self.use_checkerboarding:
            logger.info(
                f"Breaking {idx} into chunks without checkerboarding"
                f" with {self.processing_chunker}."
            )
            if self.processing_chunk_size % dst.backend.get_chunk_size(
                self.dst_resolution
            ) != IntVec3D(0, 0, 0):
                raise ValueError(
                    "`processing_chunk_size` must be evenly divisible by the `dst`"
                    f" VolumetricLayer's chunk size; received {tuple(self.processing_chunk_size)},"
                    f" which is not divisible by {dst.backend.get_chunk_size(self.dst_resolution)}"
                )
            idx_chunks = self.processing_chunker(
                idx,
                mode="exact",
                stride_start_offset=dst.backend.get_voxel_offset(self.dst_resolution),
            )
            tasks = self.make_tasks_without_checkerboarding(idx_chunks, dst, **kwargs)
            logger.info(
                f"Submitting {len(tasks)} processing tasks from operation {self.operation}."
            )
            yield tasks
        # case with checkerboarding
        else:
            if not self.max_reduction_chunk_size >= dst.backend.get_chunk_size(
                self.dst_resolution
            ):
                raise ValueError(
                    "`max_reduction_chunk_size` (which defaults to `processing_chunk_size` when"
                    " not specified)` must be at least as large as the `dst` VolumetricLayer's"
                    f" chunk size; received {self.max_reduction_chunk_size}, which is"
                    f" smaller than {dst.backend.get_chunk_size(self.dst_resolution)}"
                )
            reduction_chunker = VolumetricIndexChunker(
                chunk_size=dst.backend.get_chunk_size(self.dst_resolution),
                resolution=self.dst_resolution,
                max_superchunk_size=self.max_reduction_chunk_size,
            )
            logger.info(
                f"Breaking {idx} into reduction chunks with checkerboarding"
                f" with {reduction_chunker}. Processing chunks will use the padded index"
                f" {idx.padded(self.crop_pad)} and be chunked with {self.processing_chunker}."
            )
            red_chunks = reduction_chunker(
                idx,
                mode="expand",
                stride_start_offset=dst.backend.get_voxel_offset(self.dst_resolution),
            )
            (
                tasks,
                tasks_weights,
                red_chunks_task_idxs,
                red_chunks_temps,
                red_chunks_temps_weights,
            ) = self.make_tasks_with_checkerboarding(
                idx.padded(self.crop_pad), red_chunks, dst, **kwargs
            )
            logger.info(
                "Writing to temporary destinations:\n"
                f" Submitting {len(tasks)} processing tasks from operation {self.operation}."
            )
            yield tasks + tasks_weights
            yield mazepa.Dependency()
            tasks_reduce = [
                reduce_by_weighted_sum.make_task(
                    src_idxs=red_chunk_task_idxs,
                    src_layers=red_chunk_temps,
                    src_layers_weights=red_chunk_temps_weights,
                    idx=red_chunk,
                    dst=dst,
                    use_weights=self.use_checkerboarding_weights,
                )
                for (
                    red_chunk_task_idxs,
                    red_chunk_temps,
                    red_chunk_temps_weights,
                    red_chunk,
                ) in zip(
                    red_chunks_task_idxs, red_chunks_temps, red_chunks_temps_weights, red_chunks
                )
            ]
            logger.info(
                "Collating temporary destination CloudVolumes into the final destination:"
                f" Submitting {len(tasks_reduce)} tasks."
            )
            yield tasks_reduce


@builder.register("build_blendable_apply_flow")
def build_blendable_apply_flow(  # pylint: disable=keyword-arg-before-vararg
    operation: BlendableOpProtocol[P, R_co],
    bbox: BBox3D,
    dst_resolution: Vec3D,
    processing_chunk_size: IntVec3D,
    max_reduction_chunk_size: Optional[IntVec3D] = None,
    crop_pad: Optional[IntVec3D] = None,
    blend_pad: Optional[IntVec3D] = None,
    blend_mode: Literal["linear", "quadratic"] = "linear",
    temp_layers_dir: Optional[str] = None,
    expand: Optional[bool] = False,
    *args: P.args,
    **kwargs: P.kwargs,
) -> mazepa.Flow:
    idx = VolumetricIndex(resolution=dst_resolution, bbox=bbox)
    flow_schema: BlendableApplyFlowSchema[P, R_co] = BlendableApplyFlowSchema(
        operation=operation,
        processing_chunk_size=processing_chunk_size,
        max_reduction_chunk_size=max_reduction_chunk_size,
        dst_resolution=dst_resolution,
        crop_pad=crop_pad,
        blend_pad=blend_pad,
        blend_mode=blend_mode,
        temp_layers_dir=temp_layers_dir,
        expand=expand,
    )
    flow = flow_schema(idx, *args, **kwargs)

    return flow

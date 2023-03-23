from copy import deepcopy
from os import path
from typing import Any, Generic, Iterable, List, Literal, Optional, Tuple, TypeVar

import attrs
import torch
from typeguard import suppress_type_checks
from typing_extensions import ParamSpec

from zetta_utils import log, mazepa
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import (
    VolumetricBasedLayerProtocol,
    VolumetricIndex,
    VolumetricIndexChunker,
)
from zetta_utils.layer.volumetric.tensorstore import TSBackend
from zetta_utils.typing import check_type

from ..operation_protocols import VolumetricOpProtocol

logger = log.get_logger("zetta_utils")

IndexT = TypeVar("IndexT")
P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)

LocalBackend = TSBackend


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


@mazepa.taskable_operation_cls
@attrs.mutable
class Copy:
    def __call__(
        self,
        src: VolumetricBasedLayerProtocol,
        dst: VolumetricBasedLayerProtocol,
        idx: VolumetricIndex,
    ) -> None:
        dst[idx] = src[idx]


@mazepa.taskable_operation_cls
@attrs.mutable
class ReduceByWeightedSum:
    def __call__(
        self,
        src_idxs: List[VolumetricIndex],
        src_layers: List[VolumetricBasedLayerProtocol],
        red_idx: VolumetricIndex,
        roi_idx: VolumetricIndex,
        dst: VolumetricBasedLayerProtocol,
        processing_blend_pad: Vec3D[int],
        processing_blend_mode: Literal["linear", "quadratic"] = "linear",
    ) -> None:
        if len(src_layers) == 0:
            return
        res = torch.zeros((dst.backend.num_channels, *red_idx.shape), dtype=dst.backend.dtype)

        assert len(src_layers) > 0
        if processing_blend_pad != Vec3D[int](0, 0, 0):
            for src_idx, layer in zip(src_idxs, src_layers):
                weight = get_blending_weights(
                    idx_subchunk=src_idx,
                    idx_roi=roi_idx,
                    idx_red=red_idx,
                    processing_blend_pad=processing_blend_pad,
                    processing_blend_mode=processing_blend_mode,
                )
                intscn, subidx = get_intersection_and_subindex(src_idx, red_idx)
                subidx.insert(0, slice(0, res.shape[0]))
                res[subidx] = res[subidx] + layer[intscn] * weight
        else:
            for src_idx, layer in zip(src_idxs, src_layers):
                intscn, subidx = get_intersection_and_subindex(src_idx, red_idx)
                subidx.insert(0, slice(0, res.shape[0]))
                res[subidx] = res[subidx] + layer[intscn]
        dst[red_idx] = res


def get_blending_weights(  # pylint:disable=too-many-branches, too-many-locals
    idx_subchunk: VolumetricIndex,
    idx_roi: VolumetricIndex,
    idx_red: VolumetricIndex,
    processing_blend_pad: Vec3D[int],
    processing_blend_mode: Literal["linear", "quadratic"] = "linear",
) -> torch.Tensor:
    """
    Gets the correct blending weights for an `idx_subchunk` inside a `idx_roi` being
    reduced inside the `idx_red` region, suppressing the weights for the dimension(s)
    where `idx_subchunk` is aligned to the edge of `idx_roi`.
    """
    if processing_blend_pad == Vec3D[int](0, 0, 0):
        raise ValueError("`processing_blend_pad` must be nonzero to need blending weights")
    if not idx_subchunk.intersects(idx_roi):
        raise ValueError(
            "`idx_roi` must intersect `idx_subchunk`;"
            " `idx_roi`: {idx_roi}, `idx_subchunk`: {idx_subchunk}"
        )
    if not idx_subchunk.intersects(idx_red):
        raise ValueError(
            "`idx_red` must intersect `idx_subchunk`;"
            " `idx_red`: {idx_red}, `idx_subchunk`: {idx_subchunk}"
        )

    x_pad, y_pad, z_pad = processing_blend_pad
    intscn, _ = get_intersection_and_subindex(idx_subchunk, idx_red)
    _, intscn_in_subchunk = get_intersection_and_subindex(intscn, idx_subchunk)
    weight = torch.ones(tuple(idx_subchunk.shape), dtype=torch.float)

    if processing_blend_mode == "linear":
        weights_x = [x / (2 * x_pad) for x in range(2 * x_pad)]
        weights_y = [y / (2 * y_pad) for y in range(2 * y_pad)]
        weights_z = [z / (2 * z_pad) for z in range(2 * z_pad)]
    elif processing_blend_mode == "quadratic":
        weights_x = [
            ((x / x_pad) ** 2) / 2 if x < x_pad else 1 - ((2 - (x / x_pad)) ** 2) / 2
            for x in range(2 * x_pad)
        ]
        weights_y = [
            ((y / y_pad) ** 2) / 2 if y < y_pad else 1 - ((2 - (y / y_pad)) ** 2) / 2
            for y in range(2 * y_pad)
        ]
        weights_z = [
            ((z / z_pad) ** 2) / 2 if z < z_pad else 1 - ((2 - (z / z_pad)) ** 2) / 2
            for z in range(2 * z_pad)
        ]

    weights_x_t = torch.Tensor(weights_x).unsqueeze(-1).unsqueeze(-1)
    weights_y_t = torch.Tensor(weights_y).unsqueeze(0).unsqueeze(-1)
    weights_z_t = torch.Tensor(weights_z).unsqueeze(0).unsqueeze(0)

    if idx_subchunk.start[0] != idx_roi.start[0]:
        weight[0 : 2 * x_pad, :, :] *= weights_x_t
    if idx_subchunk.stop[0] != idx_roi.stop[0]:
        weight[(-2 * x_pad) :, :, :] *= weights_x_t.flip(0)
    if idx_subchunk.start[1] != idx_roi.start[1]:
        weight[:, 0 : 2 * y_pad, :] *= weights_y_t
    if idx_subchunk.stop[1] != idx_roi.stop[1]:
        weight[:, (-2 * y_pad) :, :] *= weights_y_t.flip(1)
    if idx_subchunk.start[2] != idx_roi.start[2]:
        weight[:, :, 0 : 2 * z_pad] *= weights_z_t
    if idx_subchunk.stop[2] != idx_roi.stop[2]:
        weight[:, :, (-2 * z_pad) :] *= weights_z_t.flip(2)

    return weight[intscn_in_subchunk].unsqueeze(0)


def set_allow_cache(*args, **kwargs):
    newargs = []
    newkwargs = {}
    for arg in args:
        if check_type(arg, VolumetricBasedLayerProtocol):
            if not arg.backend.is_local and not arg.backend.allow_cache:
                newarg = attrs.evolve(arg, backend=arg.backend.with_changes(allow_cache=True))
            else:
                newarg = arg
        newargs.append(newarg)
    for k, v in kwargs.items():
        if check_type(v, VolumetricBasedLayerProtocol):
            if not v.backend.is_local and not v.backend.allow_cache:
                newv = attrs.evolve(v, backend=v.backend.with_changes(allow_cache=True))
            else:
                newv = v
        newkwargs[k] = newv

    return newargs, newkwargs


def clear_cache(*args, **kwargs):
    for arg in args:
        if check_type(arg, VolumetricBasedLayerProtocol):
            if not arg.backend.is_local and arg.backend.allow_cache:
                arg.backend.clear_cache()
    for kwarg in kwargs.values():
        if check_type(kwarg, VolumetricBasedLayerProtocol):
            if not kwarg.backend.is_local and kwarg.backend.allow_cache:
                kwarg.backend.clear_cache()


@mazepa.flow_schema_cls
@attrs.mutable
class VolumetricApplyFlowSchema(Generic[P, R_co]):
    op: VolumetricOpProtocol[P, Any, Any]
    processing_chunk_size: Vec3D[int]
    dst_resolution: Vec3D
    max_reduction_chunk_size: Optional[Vec3D[int]] = None
    max_reduction_chunk_size_final: Vec3D[int] = attrs.field(init=False)
    roi_crop_pad: Optional[Vec3D[int]] = None
    processing_blend_pad: Optional[Vec3D[int]] = None
    processing_blend_mode: Literal["linear", "quadratic"] = "linear"
    intermediaries_dir: Optional[str] = None
    allow_cache: bool = False
    clear_cache_on_return: bool = False
    force_intermediaries: bool = False
    use_checkerboarding: bool = attrs.field(init=False)
    processing_chunker: VolumetricIndexChunker = attrs.field(init=False)

    def _get_backend_chunk_size_to_use(self, dst) -> Vec3D[int]:
        backend_chunk_size = deepcopy(self.processing_chunk_size)
        dst_backend_chunk_size = dst.backend.get_chunk_size(self.dst_resolution)
        for i in range(3):
            if backend_chunk_size[i] == 1:
                continue
            if self.processing_chunk_size[i] % 2 != 0:
                raise ValueError(
                    "`processing_chunk_size` must be divisible by 2 at least once in"
                    " dimensions that are not 1;"
                    " received {self.processing_chunk_size[i]} against {dst_backend_chunk_size[i]}"
                    # turn caching back off
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
        if self.roi_crop_pad is None:
            self.roi_crop_pad = Vec3D[int](0, 0, 0)
        if self.processing_blend_pad is None:
            self.processing_blend_pad = Vec3D[int](0, 0, 0)
        if self.roi_crop_pad != Vec3D[int](0, 0, 0) or self.processing_blend_pad != Vec3D[int](
            0, 0, 0
        ):
            self.use_checkerboarding = True
        else:
            self.use_checkerboarding = False

        if self.use_checkerboarding:
            if not self.processing_blend_pad <= self.processing_chunk_size // 2:
                raise ValueError(
                    f" `processing_blend_pad` must be less than or equal to"
                    f" half of `processing_chunk_size`; received {self.processing_blend_pad}",
                    f" which is larger than {self.processing_size // 2}",
                )
            if self.intermediaries_dir is None:
                raise ValueError(
                    "`intermediaries_dir` must be specified when using blending or crop."
                )
        if self.force_intermediaries:
            if self.intermediaries_dir is None:
                raise ValueError(
                    "`intermediaries_dir` must be specified when `force_intermediaries`==True."
                )

        self.processing_chunker = VolumetricIndexChunker(
            chunk_size=self.processing_chunk_size, resolution=self.dst_resolution
        )
        if self.max_reduction_chunk_size is None:
            self.max_reduction_chunk_size_final = self.processing_chunk_size
        else:
            self.max_reduction_chunk_size_final = self.max_reduction_chunk_size

    def _get_temp_dst(
        self,
        dst: VolumetricBasedLayerProtocol,
        idx: VolumetricIndex,
        suffix: Optional[Any] = None,
    ) -> VolumetricBasedLayerProtocol:
        assert self.intermediaries_dir is not None
        temp_name = f"_{self.op.__class__.__name__}_temp_{idx.pformat()}_{suffix}"
        allow_cache = self.allow_cache and not self.intermediaries_dir.startswith("file://")
        backend_chunk_size = self._get_backend_chunk_size_to_use(dst)
        if self.intermediaries_dir.startswith("file://"):
            backend_temp = dst.backend.as_type(LocalBackend).with_changes(
                name=path.join(self.intermediaries_dir, temp_name),
                voxel_offset_res=(idx.start - backend_chunk_size, self.dst_resolution),
                chunk_size_res=(backend_chunk_size, self.dst_resolution),
                enforce_chunk_aligned_writes=False,
                allow_cache=allow_cache,
            )
        else:
            backend_temp = dst.backend.with_changes(
                name=path.join(self.intermediaries_dir, temp_name),
                voxel_offset_res=(idx.start - backend_chunk_size, self.dst_resolution),
                chunk_size_res=(backend_chunk_size, self.dst_resolution),
                enforce_chunk_aligned_writes=False,
                use_compression=False,
                allow_cache=allow_cache,
            )
        return attrs.evolve(deepcopy(dst), backend=backend_temp)

    def make_tasks_without_checkerboarding(
        self,
        idx_chunks: Iterable[VolumetricIndex],
        dst: VolumetricBasedLayerProtocol,
        **kwargs: P.kwargs,
    ) -> List[mazepa.tasks.Task[R_co]]:
        tasks = [
            self.op.make_task(
                idx=idx_chunk,
                dst=dst,
                **kwargs,
            )
            for idx_chunk in idx_chunks
        ]

        return tasks

    def make_tasks_with_intermediaries(  # pylint: disable=too-many-locals
        self,
        idx: VolumetricIndex,
        dst: VolumetricBasedLayerProtocol,
        **kwargs: P.kwargs,
    ) -> Tuple[List[mazepa.tasks.Task[R_co]], VolumetricBasedLayerProtocol]:
        dst_temp = self._get_temp_dst(dst, idx)
        idx_chunks = self.processing_chunker(idx, mode="exact")
        tasks = self.make_tasks_without_checkerboarding(idx_chunks, dst_temp, **kwargs)

        return tasks, dst_temp

    def make_tasks_with_checkerboarding(  # pylint: disable=too-many-locals
        self,
        idx: VolumetricIndex,
        red_chunks: Iterable[VolumetricIndex],
        dst: VolumetricBasedLayerProtocol,
        **kwargs: P.kwargs,
    ) -> Tuple[
        List[mazepa.tasks.Task[R_co]],
        List[List[VolumetricIndex]],
        List[List[VolumetricBasedLayerProtocol]],
        List[VolumetricBasedLayerProtocol],
    ]:
        assert self.intermediaries_dir is not None
        assert self.processing_blend_pad is not None
        """
        Makes tasks that can be reduced to the final output, tasks that write weights for the
        reduction, and for each reduction chunk, the list of task indices, the list of temporary
        output layers, and the list of temporary layers with weights for all tasks that intersect
        the reduction chunk that need to be reduced for the output.
        """
        tasks: List[mazepa.tasks.Task[R_co]] = []
        red_chunks_task_idxs: List[List[VolumetricIndex]] = [[] for _ in red_chunks]
        red_chunks_temps: List[List[VolumetricBasedLayerProtocol]] = [[] for _ in red_chunks]
        dst_temps: List[VolumetricBasedLayerProtocol] = []
        for chunker, chunker_idx in self.processing_chunker.split_into_nonoverlapping_chunkers(
            self.processing_blend_pad
        ):
            """
            Prepare the temporary destination by allowing non-aligned writes, aligning the voxel
            offset of the temporary destination with where the given idx starts, and setting the
            backend chunk size to half of the processing chunk size. Furthermore, skip caching
            if the temporary destination happens to be local.
            """
            dst_temp = self._get_temp_dst(dst, idx, chunker_idx)
            dst_temps.append(dst_temp)

            # assert that the idx passed in is in fact exactly divisible by the chunk size
            red_chunk_aligned = idx.snapped(
                grid_offset=idx.start, grid_size=self.processing_chunk_size, mode="expand"
            )
            if red_chunk_aligned != idx:
                raise ValueError(
                    f"received (crop padded) idx {idx} is not evenly divisible by"
                    f" {self.processing_chunk_size}"
                )
            # expand to allow for processing_blend_pad around the edges
            idx_expanded = idx.padded(self.processing_blend_pad)

            task_idxs = chunker(
                idx_expanded,
                stride_start_offset_in_unit=idx_expanded.start * idx_expanded.resolution,
                mode="shrink",
            )
            with suppress_type_checks():
                for task_idx in task_idxs:
                    tasks.append(self.op.make_task(task_idx, dst_temp, **kwargs))
                    for i, red_chunk in enumerate(red_chunks):
                        if task_idx.intersects(red_chunk):
                            red_chunks_task_idxs[i].append(task_idx)
                            red_chunks_temps[i].append(dst_temp)

        return (tasks, red_chunks_task_idxs, red_chunks_temps, dst_temps)

    def flow(  # pylint:disable=too-many-branches, too-many-statements
        self,
        idx: VolumetricIndex,
        dst: VolumetricBasedLayerProtocol,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> mazepa.FlowFnReturnType:
        assert len(args) == 0
        assert self.roi_crop_pad is not None
        assert self.processing_blend_pad is not None

        # set caching for all VolumetricBasedLayerProtocols as desired
        if self.allow_cache:
            args, kwargs = set_allow_cache(*args, **kwargs)

        logger.info(f"Breaking {idx} into chunks with {self.processing_chunker}.")

        # case without checkerboarding
        if not self.use_checkerboarding and not self.force_intermediaries:
            idx_chunks = self.processing_chunker(idx, mode="exact")
            tasks = self.make_tasks_without_checkerboarding(idx_chunks, dst, **kwargs)
            logger.info(f"Submitting {len(tasks)} processing tasks from operation {self.op}.")
            yield tasks
        elif not self.use_checkerboarding and self.force_intermediaries:
            tasks, dst_temp = self.make_tasks_with_intermediaries(idx, dst, **kwargs)
            logger.info(f"Submitting {len(tasks)} processing tasks from operation {self.op}.")
            yield tasks
            yield mazepa.Dependency()
            if not self.max_reduction_chunk_size_final >= dst.backend.get_chunk_size(
                self.dst_resolution
            ):
                copy_chunk_size = dst.backend.get_chunk_size(self.dst_resolution)
            else:
                copy_chunk_size = self.max_reduction_chunk_size_final
            reduction_chunker = VolumetricIndexChunker(
                chunk_size=dst.backend.get_chunk_size(self.dst_resolution),
                resolution=self.dst_resolution,
                max_superchunk_size=copy_chunk_size,
            )
            logger.info(
                f"Breaking {idx} into chunks to be copied from the intermediary layer"
                f" with {reduction_chunker}."
            )
            stride_start_offset = dst.backend.get_voxel_offset(self.dst_resolution)
            stride_start_offset_in_unit = stride_start_offset * self.dst_resolution
            red_chunks = reduction_chunker(
                idx, mode="exact", stride_start_offset_in_unit=stride_start_offset_in_unit
            )
            tasks_reduce = [
                Copy().make_task(src=dst_temp, dst=dst, idx=red_chunk) for red_chunk in red_chunks
            ]
            logger.info(
                "Copying temporary destination CloudVolume into the final destination:"
                f" Submitting {len(tasks_reduce)} tasks."
            )
            yield tasks_reduce
            clear_cache(dst_temp)
        # case with checkerboarding
        else:
            if dst.backend.enforce_chunk_aligned_writes:
                try:
                    dst.backend.assert_idx_is_chunk_aligned(idx)
                except Exception as e:
                    error_str = (
                        "`dst` VolumetricBasedLayerProtocol's backend has"
                        " `enforce_chunk_aligned_writes`=True, but the provided `idx`"
                        " is not chunk aligned:\n"
                    )
                    e.args = (error_str + e.args[0],)
                    raise e
            if not self.max_reduction_chunk_size_final >= dst.backend.get_chunk_size(
                self.dst_resolution
            ):
                raise ValueError(
                    "`max_reduction_chunk_size` (which defaults to `processing_chunk_size` when"
                    " not specified)` must be at least as large as the `dst` layer's"
                    f" chunk size; received {self.max_reduction_chunk_size_final}, which is"
                    f" smaller than {dst.backend.get_chunk_size(self.dst_resolution)}"
                )
            reduction_chunker = VolumetricIndexChunker(
                chunk_size=dst.backend.get_chunk_size(self.dst_resolution),
                resolution=self.dst_resolution,
                max_superchunk_size=self.max_reduction_chunk_size_final,
            )
            logger.info(
                f"Breaking {idx} into reduction chunks with checkerboarding"
                f" with {reduction_chunker}. Processing chunks will use the padded index"
                f" {idx.padded(self.roi_crop_pad)} and be chunked with {self.processing_chunker}."
            )
            stride_start_offset = dst.backend.get_voxel_offset(self.dst_resolution)
            stride_start_offset_in_unit = stride_start_offset * self.dst_resolution
            red_chunks = reduction_chunker(
                idx, mode="exact", stride_start_offset_in_unit=stride_start_offset_in_unit
            )
            (
                tasks,
                red_chunks_task_idxs,
                red_chunks_temps,
                dst_temps,
            ) = self.make_tasks_with_checkerboarding(
                idx.padded(self.roi_crop_pad), red_chunks, dst, **kwargs
            )
            logger.info(
                "Writing to temporary destinations:\n"
                f" Submitting {len(tasks)} processing tasks from operation {self.op}."
            )
            yield tasks
            yield mazepa.Dependency()
            tasks_reduce = [
                ReduceByWeightedSum().make_task(
                    src_idxs=red_chunk_task_idxs,
                    src_layers=red_chunk_temps,
                    red_idx=red_chunk,
                    roi_idx=idx.padded(self.roi_crop_pad + self.processing_blend_pad),
                    dst=dst,
                    processing_blend_pad=self.processing_blend_pad,
                    processing_blend_mode=self.processing_blend_mode,
                )
                for (
                    red_chunk_task_idxs,
                    red_chunk_temps,
                    red_chunk,
                ) in zip(red_chunks_task_idxs, red_chunks_temps, red_chunks)
            ]
            logger.info(
                "Collating temporary destination CloudVolumes into the final destination:"
                f" Submitting {len(tasks_reduce)} tasks."
            )
            yield tasks_reduce
            clear_cache(*dst_temps)
        if self.clear_cache_on_return:
            clear_cache(*args, **kwargs)


def build_volumetric_apply_flow(  # pylint: disable=keyword-arg-before-vararg
    op: VolumetricOpProtocol[P, R_co, Any],
    start_coord: Vec3D[int],
    end_coord: Vec3D[int],
    coord_resolution: Vec3D,
    dst_resolution: Vec3D,
    processing_chunk_size: Vec3D[int],
    processing_crop_pad: Vec3D[int],
    max_reduction_chunk_size: Optional[Vec3D[int]] = None,
    roi_crop_pad: Optional[Vec3D[int]] = None,
    processing_blend_pad: Optional[Vec3D[int]] = None,
    processing_blend_mode: Literal["linear", "quadratic"] = "linear",
    intermediaries_dir: Optional[str] = None,
    allow_cache: bool = False,
    clear_cache_on_return: bool = False,
    *args: P.args,
    **kwargs: P.kwargs,
) -> mazepa.Flow:
    bbox: BBox3D = BBox3D.from_coords(
        start_coord=start_coord, end_coord=end_coord, resolution=coord_resolution
    )
    idx = VolumetricIndex(resolution=dst_resolution, bbox=bbox)
    flow_schema: VolumetricApplyFlowSchema[P, R_co] = VolumetricApplyFlowSchema(
        op=op.with_added_crop_pad(processing_crop_pad),
        processing_chunk_size=processing_chunk_size,
        max_reduction_chunk_size=max_reduction_chunk_size,
        dst_resolution=dst_resolution,
        roi_crop_pad=roi_crop_pad,
        processing_blend_pad=processing_blend_pad,
        processing_blend_mode=processing_blend_mode,
        intermediaries_dir=intermediaries_dir,
        allow_cache=allow_cache,
        clear_cache_on_return=clear_cache_on_return,
    )
    flow = flow_schema(idx, *args, **kwargs)

    return flow

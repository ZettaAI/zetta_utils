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
from zetta_utils.typing import check_type

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


@mazepa.taskable_operation_cls
@attrs.mutable
class ReduceByWeightedSum:
    def __call__(
        self,
        src_idxs: List[VolumetricIndex],
        src_layers: List[VolumetricLayer],
        red_idx: VolumetricIndex,
        fov_idx: VolumetricIndex,
        dst: VolumetricLayer,
        processing_blend_pad: IntVec3D = IntVec3D(0, 0, 0),
        processing_blend_mode: Literal["linear", "quadratic"] = "linear",
    ) -> None:
        if len(src_layers) == 0:
            return
        res = torch.zeros((dst.backend.num_channels, *red_idx.shape), dtype=dst.backend.dtype)

        assert len(src_layers) > 0
        if processing_blend_pad != IntVec3D(0, 0, 0):
            for src_idx, layer in zip(src_idxs, src_layers):
                weight = get_blending_weights(
                    idx_subchunk=src_idx,
                    idx_fov=fov_idx,
                    idx_red=red_idx,
                    processing_blend_pad=processing_blend_pad,
                    processing_blend_mode=processing_blend_mode,
                )
                intscn, subidx = get_intersection_and_subindex(src_idx, red_idx)
                subidx.insert(0, slice(0, res.shape[0]))
                res[subidx] = tensor_ops.common.add(res[subidx], layer[intscn] * weight)
        else:
            for src_idx, layer in zip(src_idxs, src_layers):
                intscn, subidx = get_intersection_and_subindex(src_idx, red_idx)
                subidx.insert(0, slice(0, res.shape[0]))
                res[subidx] = tensor_ops.common.add(res[subidx], layer[intscn])
        dst[red_idx] = res


def get_blending_weights(  # pylint:disable=too-many-branches, too-many-locals
    idx_subchunk: VolumetricIndex,
    idx_fov: VolumetricIndex,
    idx_red: VolumetricIndex,
    processing_blend_pad: IntVec3D,
    processing_blend_mode: Literal["linear", "quadratic"] = "linear",
) -> torch.Tensor:
    """
    Gets the correct blending weights for an `idx_subchunk` inside a `idx_fov` being
    reduced inside the `idx_red` region, suppressing the weights for the dimension(s)
    where `idx_subchunk` is aligned to the edge of `idx_fov`.
    """
    if processing_blend_pad == IntVec3D(0, 0, 0):
        raise ValueError("`processing_blend_pad` must be nonzero to need blending weights")
    if not idx_subchunk.intersects(idx_fov):
        raise ValueError(
            "`idx_fov` must intersect `idx_subchunk`;"
            " `idx_fov`: {idx_fov}, `idx_subchunk`: {idx_subchunk}"
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

    if idx_subchunk.start[0] != idx_fov.start[0]:
        weight[0 : 2 * x_pad, :, :] *= weights_x_t
    if idx_subchunk.stop[0] != idx_fov.stop[0]:
        weight[(-2 * x_pad) :, :, :] *= weights_x_t.flip(0)
    if idx_subchunk.start[1] != idx_fov.start[1]:
        weight[:, 0 : 2 * y_pad, :] *= weights_y_t
    if idx_subchunk.stop[1] != idx_fov.stop[1]:
        weight[:, (-2 * y_pad) :, :] *= weights_y_t.flip(1)
    if idx_subchunk.start[2] != idx_fov.start[2]:
        weight[:, :, 0 : 2 * z_pad] *= weights_z_t
    if idx_subchunk.stop[2] != idx_fov.stop[2]:
        weight[:, :, (-2 * z_pad) :] *= weights_z_t.flip(2)

    return weight[intscn_in_subchunk].unsqueeze(0)


def set_allow_cache(*args, **kwargs):
    newargs = []
    newkwargs = {}
    for arg in args:
        if check_type(arg, VolumetricLayer):
            if not arg.backend.is_local and not arg.backend.allow_cache:
                newarg = attrs.evolve(arg, backend=arg.backend.with_changes(allow_cache=True))
            else:
                newarg = arg
        newargs.append(newarg)
    for k, v in kwargs.items():
        if check_type(v, VolumetricLayer):
            if not v.backend.is_local and not v.backend.allow_cache:
                newv = attrs.evolve(v, backend=v.backend.with_changes(allow_cache=True))
            else:
                newv = v
        newkwargs[k] = newv

    return newargs, newkwargs


def clear_cache(*args, **kwargs):
    for arg in args:
        if check_type(arg, VolumetricLayer):
            if not arg.backend.is_local and arg.backend.allow_cache:
                arg.backend.clear_cache()
    for kwarg in kwargs.values():
        if check_type(kwarg, VolumetricLayer):
            if not kwarg.backend.is_local and kwarg.backend.allow_cache:
                kwarg.backend.clear_cache()


@builder.register("BlendableApplyFlowSchema")
@mazepa.flow_schema_cls
@attrs.mutable
class BlendableApplyFlowSchema(Generic[P, R_co]):
    op: BlendableOpProtocol[P, R_co]
    processing_chunk_size: IntVec3D
    dst_resolution: Vec3D
    max_reduction_chunk_size: Optional[IntVec3D] = None
    fov_crop_pad: Optional[IntVec3D] = None
    processing_blend_pad: Optional[IntVec3D] = None
    processing_blend_mode: Literal["linear", "quadratic"] = "linear"
    temp_layers_dir: Optional[str] = None
    allow_cache: bool = False
    clear_cache_on_return: bool = False
    use_checkerboarding: bool = attrs.field(init=False)
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
        if self.fov_crop_pad is None:
            self.fov_crop_pad = IntVec3D(0, 0, 0)
        if self.processing_blend_pad is None:
            self.processing_blend_pad = IntVec3D(0, 0, 0)
        if self.fov_crop_pad != IntVec3D(0, 0, 0) or self.processing_blend_pad != IntVec3D(
            0, 0, 0
        ):
            self.use_checkerboarding = True
        else:
            # even if fov_crop_pad and processing_blend_pad are both zero,
            # use checkerboarding if max_reduction_chunk is set
            if self.max_reduction_chunk_size is not None:
                self.use_checkerboarding = True
                logger.info(
                    "Using checkerboarding even though `fov_crop_pad` "
                    " and `processing_blend_pad` are zero"
                    " since `max_reduction_chunk_size` is nonzero;"
                    " received {self.max_reduction_chunk_size}"
                )
            else:
                self.use_checkerboarding = False
        if self.use_checkerboarding:
            if not self.processing_blend_pad <= self.processing_chunk_size // 2:
                raise ValueError(
                    f" `processing_blend_pad` must be less than or equal to"
                    f" half of `processing_chunk_size`; received {self.processing_blend_pad}",
                    f" which is larger than {self.processing_size // 2}",
                )
            if self.temp_layers_dir is None:
                raise ValueError("`temp_layers_dir` must be specified when using blending or crop")
        self.processing_chunker = VolumetricIndexChunker(
            chunk_size=self.processing_chunk_size, resolution=self.dst_resolution
        )
        if self.max_reduction_chunk_size is None:
            self.max_reduction_chunk_size = self.processing_chunk_size

    def make_tasks_without_checkerboarding(
        self, idx_chunks: Iterable[VolumetricIndex], dst: VolumetricLayer, **kwargs: P.kwargs
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

    def make_tasks_with_checkerboarding(  # pylint: disable=too-many-locals
        self,
        idx: VolumetricIndex,
        red_chunks: Iterable[VolumetricIndex],
        dst: VolumetricLayer,
        **kwargs: P.kwargs,
    ) -> Tuple[
        List[mazepa.tasks.Task[R_co]],
        List[List[VolumetricIndex]],
        List[List[VolumetricLayer]],
        List[VolumetricLayer],
    ]:
        assert self.temp_layers_dir is not None
        assert self.processing_blend_pad is not None
        """
        Makes tasks that can be reduced to the final output, tasks that write weights for the
        reduction, and for each reduction chunk, the list of task indices, the list of temporary
        output layers, and the list of temporary layers with weights for all tasks that intersect
        the reduction chunk that need to be reduced for the output.
        """
        tasks: List[mazepa.tasks.Task[R_co]] = []
        red_chunks_task_idxs: List[List[VolumetricIndex]] = [[] for _ in red_chunks]
        red_chunks_temps: List[List[VolumetricLayer]] = [[] for _ in red_chunks]
        dst_temps: List[VolumetricLayer] = []
        for chunker, chunker_idx in self.processing_chunker.split_into_nonoverlapping_chunkers(
            self.processing_blend_pad
        ):
            """
            Prepare the temporary destination by allowing non-aligned writes, aligning the voxel
            offset of the temporary destination with where the given idx starts, and setting the
            backend chunk size to half of the processing chunk size. Furthermore, skip caching
            if the temporary destination happens to be local.
            """
            allow_cache = self.allow_cache and not self.temp_layers_dir.startswith("file://")
            backend_temp = dst.backend.with_changes(
                name=path.join(
                    self.temp_layers_dir,
                    f"_{self.op.__class__.__name__}_temp_{idx.pformat()}_{chunker_idx}",
                ),
                voxel_offset_res=(idx.start, self.dst_resolution),
                chunk_size_res=(self._get_backend_chunk_size_to_use(dst), self.dst_resolution),
                enforce_chunk_aligned_writes=False,
                use_compression=False,
                allow_cache=allow_cache,
            )
            dst_temp = attrs.evolve(deepcopy(dst), backend=backend_temp)
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

            task_idxs = chunker(idx_expanded, stride_start_offset=idx_expanded.start)

            for task_idx in task_idxs:
                tasks.append(self.op.make_task(task_idx, dst_temp, **kwargs))
                for i, red_chunk in enumerate(red_chunks):
                    if task_idx.intersects(red_chunk):
                        red_chunks_task_idxs[i].append(task_idx)
                        red_chunks_temps[i].append(dst_temp)

        return (tasks, red_chunks_task_idxs, red_chunks_temps, dst_temps)

    def flow(  # pylint:disable=too-many-branches
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> mazepa.FlowFnReturnType:
        assert self.fov_crop_pad is not None
        assert self.processing_blend_pad is not None

        # set caching for all VolumetricLayers as desired
        if self.allow_cache:
            args, kwargs = set_allow_cache(*args, **kwargs)

        logger.info(f"Breaking {idx} into chunks with {self.processing_chunker}.")

        # case without checkerboarding
        if not self.use_checkerboarding:
            dst.backend.assert_idx_is_chunk_aligned(idx)
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
            logger.info(f"Submitting {len(tasks)} processing tasks from operation {self.op}.")
            yield tasks
        # case with checkerboarding
        else:
            if dst.backend.enforce_chunk_aligned_writes:
                try:
                    dst.backend.assert_idx_is_chunk_aligned(idx)
                except Exception as e:
                    error_str = (
                        "`dst` VolumetricLayer's backend has"
                        " `enforce_chunk_aligned_writes`=True, but the provided `idx`"
                        " is not chunk aligned:\n"
                    )
                    e.args = (error_str + e.args[0],)
                    raise e
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
                f" {idx.padded(self.fov_crop_pad)} and be chunked with {self.processing_chunker}."
            )
            red_chunks = reduction_chunker(
                idx,
                mode="exact",
                stride_start_offset=dst.backend.get_voxel_offset(self.dst_resolution),
            )
            (
                tasks,
                red_chunks_task_idxs,
                red_chunks_temps,
                dst_temps,
            ) = self.make_tasks_with_checkerboarding(
                idx.padded(self.fov_crop_pad), red_chunks, dst, **kwargs
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
                    fov_idx=idx.padded(self.fov_crop_pad + self.processing_blend_pad),
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


@builder.register("build_blendable_apply_flow")
def build_blendable_apply_flow(  # pylint: disable=keyword-arg-before-vararg
    op: BlendableOpProtocol[P, R_co],
    start_coord: IntVec3D,
    end_coord: IntVec3D,
    coord_resolution: Vec3D,
    dst_resolution: Vec3D,
    processing_chunk_size: IntVec3D,
    processing_crop_pad: IntVec3D = IntVec3D(0, 0, 0),
    max_reduction_chunk_size: Optional[IntVec3D] = None,
    fov_crop_pad: Optional[IntVec3D] = None,
    processing_blend_pad: Optional[IntVec3D] = None,
    processing_blend_mode: Literal["linear", "quadratic"] = "linear",
    temp_layers_dir: Optional[str] = None,
    allow_cache: bool = False,
    clear_cache_on_return: bool = False,
    *args: P.args,
    **kwargs: P.kwargs,
) -> mazepa.Flow:
    bbox: BBox3D = BBox3D.from_coords(
        start_coord=start_coord, end_coord=end_coord, resolution=coord_resolution
    )
    idx = VolumetricIndex(resolution=dst_resolution, bbox=bbox)
    flow_schema: BlendableApplyFlowSchema[P, R_co] = BlendableApplyFlowSchema(
        op=op.with_added_crop_pad(processing_crop_pad),
        processing_chunk_size=processing_chunk_size,
        max_reduction_chunk_size=max_reduction_chunk_size,
        dst_resolution=dst_resolution,
        fov_crop_pad=fov_crop_pad,
        processing_blend_pad=processing_blend_pad,
        processing_blend_mode=processing_blend_mode,
        temp_layers_dir=temp_layers_dir,
        allow_cache=allow_cache,
        clear_cache_on_return=clear_cache_on_return,
    )
    flow = flow_schema(idx, *args, **kwargs)

    return flow

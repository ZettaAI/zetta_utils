from __future__ import annotations

import itertools
import multiprocessing
from abc import ABC
from copy import deepcopy
from os import path
from typing import Any, Generic, List, Literal, Optional, Tuple, TypeVar

import attrs
import cachetools
import fsspec
import numpy as np
import torch
from typeguard import suppress_type_checks
from typing_extensions import ParamSpec

from zetta_utils import MULTIPROCESSING_NUM_TASKS_THRESHOLD, log, mazepa
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import (
    VolumetricBasedLayerProtocol,
    VolumetricIndex,
    VolumetricIndexChunker,
)
from zetta_utils.mazepa import semaphore
from zetta_utils.tensor_ops import convert

from ..operation_protocols import VolumetricOpProtocol

_weights_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=16)

logger = log.get_logger("zetta_utils")

IndexT = TypeVar("IndexT")
P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)


@mazepa.taskable_operation_cls
@attrs.mutable
class Copy:
    def __call__(
        self,
        src: VolumetricBasedLayerProtocol,
        dst: VolumetricBasedLayerProtocol,
        idx: VolumetricIndex,
    ) -> None:
        with semaphore("read"):
            data = src[idx]
        with semaphore("write"):
            dst[idx] = data


@mazepa.taskable_operation_cls
class ReduceOperation(ABC):
    """Base class for Reduce operations, which combine values from different
    chunks where they overlap."""

    def __call__(
        self,
        src_idxs: List[VolumetricIndex],
        src_layers: List[VolumetricBasedLayerProtocol],
        red_idx: VolumetricIndex,
        roi_idx: VolumetricIndex,
        dst: VolumetricBasedLayerProtocol,
        processing_blend_pad: Vec3D[int],
    ) -> None:
        pass


@mazepa.taskable_operation_cls
@attrs.frozen
class ReduceNaive(ReduceOperation):
    """A reducer that simply takes the maximum value at each location in the overlap area."""

    def __call__(
        self,
        src_idxs: List[VolumetricIndex],
        src_layers: List[VolumetricBasedLayerProtocol],
        red_idx: VolumetricIndex,
        roi_idx: VolumetricIndex,
        dst: VolumetricBasedLayerProtocol,
        processing_blend_pad: Vec3D[int],
    ) -> None:
        with suppress_type_checks():
            if len(src_layers) == 0:
                return
            res = np.zeros(
                (dst.backend.num_channels, *red_idx.shape),
                dtype=dst.backend.dtype,
            )
            assert len(src_layers) > 0
            if processing_blend_pad != Vec3D[int](0, 0, 0):
                for src_idx, layer in zip(src_idxs, src_layers):
                    intscn, subidx = src_idx.get_intersection_and_subindex(red_idx)
                    subidx_channels = (slice(0, res.shape[0]), *subidx)
                    with semaphore("read"):
                        res[subidx_channels] = np.maximum(res[subidx_channels], layer[intscn])
            else:
                for src_idx, layer in zip(src_idxs, src_layers):
                    intscn, subidx = src_idx.get_intersection_and_subindex(red_idx)
                    subidx_channels = (slice(0, res.shape[0]), *subidx)
                    with semaphore("read"):
                        res[subidx_channels] = layer[intscn]
            with semaphore("write"):
                dst[red_idx] = res


def is_floating_point_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.floating)


@mazepa.taskable_operation_cls
@attrs.mutable
class ReduceByWeightedSum(ReduceOperation):
    """Reducer that combines the values in the overlap area by a linear or
    quadratic function of the distance from the edge."""

    processing_blend_mode: Literal["linear", "quadratic"] = "linear"

    def __init__(self, processing_blend_mode: Literal["linear", "quadratic"]):
        self.processing_blend_mode = processing_blend_mode

    def __call__(
        self,
        src_idxs: List[VolumetricIndex],
        src_layers: List[VolumetricBasedLayerProtocol],
        red_idx: VolumetricIndex,
        roi_idx: VolumetricIndex,
        dst: VolumetricBasedLayerProtocol,
        processing_blend_pad: Vec3D[int],
    ) -> None:
        with suppress_type_checks():
            if len(src_layers) == 0:
                return
            if not is_floating_point_dtype(dst.backend.dtype) and processing_blend_pad != Vec3D[
                int
            ](0, 0, 0):
                # backend is integer, but blending is requested - need to use float to avoid
                # rounding errors
                res = torch.zeros((dst.backend.num_channels, *red_idx.shape), dtype=torch.float)
            else:
                res = torch.zeros(
                    (dst.backend.num_channels, *red_idx.shape),
                    dtype=convert.to_torch_dtype(dst.backend.dtype),
                )
            assert len(src_layers) > 0
            if processing_blend_pad != Vec3D[int](0, 0, 0):
                for src_idx, layer in zip(src_idxs, src_layers):
                    weight = get_blending_weights(
                        idx_subchunk=src_idx,
                        idx_roi=roi_idx,
                        idx_red=red_idx,
                        processing_blend_pad=processing_blend_pad,
                        processing_blend_mode=self.processing_blend_mode,
                    )
                    intscn, subidx = src_idx.get_intersection_and_subindex(red_idx)
                    subidx_channels = [slice(0, res.shape[0])] + list(subidx)
                    with semaphore("read"):
                        if not is_floating_point_dtype(dst.backend.dtype):
                            # Temporarily convert integer cutout to float for rounding
                            res[subidx_channels] = (
                                res[subidx_channels] + layer[intscn].astype(float) * weight.numpy()
                            )
                        else:
                            res[subidx_channels] = (
                                res[subidx_channels] + layer[intscn] * weight.numpy()
                            )

                if not is_floating_point_dtype(dst.backend.dtype):
                    res = res.round().to(dtype=convert.to_torch_dtype(dst.backend.dtype))
            else:
                for src_idx, layer in zip(src_idxs, src_layers):
                    intscn, subidx = src_idx.get_intersection_and_subindex(red_idx)
                    subidx_channels = [slice(0, res.shape[0])] + list(subidx)
                    with semaphore("read"):
                        res.numpy()[tuple(subidx_channels)] = layer[intscn]
            with semaphore("write"):
                dst[red_idx] = res


@cachetools.cached(_weights_cache)
def get_weight_template(
    processing_blend_mode: Literal["linear", "quadratic"],
    subchunk_shape: Tuple[int, ...],
    x_pad: int,
    y_pad: int,
    z_pad: int,
    x_start_aligned: bool,
    x_stop_aligned: bool,
    y_start_aligned: bool,
    y_stop_aligned: bool,
    z_start_aligned: bool,
    z_stop_aligned: bool,
) -> torch.Tensor:
    weight = torch.ones(subchunk_shape, dtype=torch.float)
    if processing_blend_mode == "linear":
        weights_x = [x / (2 * x_pad + 1) for x in range(1, 2 * x_pad + 1)]
        weights_y = [y / (2 * y_pad + 1) for y in range(1, 2 * y_pad + 1)]
        weights_z = [z / (2 * z_pad + 1) for z in range(1, 2 * z_pad + 1)]
    elif processing_blend_mode == "quadratic":
        weights_x = [
            (
                ((x / (x_pad + 0.5)) ** 2) / 2
                if x <= x_pad
                else 1 - ((2 - (x / (x_pad + 0.5))) ** 2) / 2
            )
            for x in range(1, 2 * x_pad + 1)
        ]
        weights_y = [
            (
                ((y / (y_pad + 0.5)) ** 2) / 2
                if y <= y_pad
                else 1 - ((2 - (y / (y_pad + 0.5))) ** 2) / 2
            )
            for y in range(1, 2 * y_pad + 1)
        ]
        weights_z = [
            (
                ((z / (z_pad + 0.5)) ** 2) / 2
                if z <= z_pad
                else 1 - ((2 - (z / (z_pad + 0.5))) ** 2) / 2
            )
            for z in range(1, 2 * z_pad + 1)
        ]

    weights_x_t = torch.Tensor(weights_x).unsqueeze(-1).unsqueeze(-1)
    weights_y_t = torch.Tensor(weights_y).unsqueeze(0).unsqueeze(-1)
    weights_z_t = torch.Tensor(weights_z).unsqueeze(0).unsqueeze(0)

    if not x_start_aligned and x_pad != 0:
        weight[0 : 2 * x_pad, :, :] *= weights_x_t
    if not x_stop_aligned and x_pad != 0:
        weight[(-2 * x_pad) :, :, :] *= weights_x_t.flip(0)
    if not y_start_aligned and y_pad != 0:
        weight[:, 0 : 2 * y_pad, :] *= weights_y_t
    if not y_stop_aligned and y_pad != 0:
        weight[:, (-2 * y_pad) :, :] *= weights_y_t.flip(1)
    if not z_start_aligned and z_pad != 0:
        weight[:, :, 0 : 2 * z_pad] *= weights_z_t
    if not z_stop_aligned and z_pad != 0:
        weight[:, :, (-2 * z_pad) :] *= weights_z_t.flip(2)
    return weight


def get_blending_weights(  # pylint:disable=too-many-branches, too-many-locals
    idx_subchunk: VolumetricIndex,
    idx_roi: VolumetricIndex,
    idx_red: VolumetricIndex,
    processing_blend_pad: Vec3D[int],
    processing_blend_mode: Literal["linear", "quadratic"],
) -> torch.Tensor:
    """
    Gets the correct blending weights for an `idx_subchunk` inside a `idx_roi` being
    reduced inside the `idx_red` region, suppressing the weights for the dimension(s)
    where `idx_subchunk` is aligned to the edge of `idx_roi`.
    """
    if processing_blend_pad == Vec3D[int](0, 0, 0):
        raise ValueError("`processing_blend_pad` must be nonzero to need blending weights")
    if not idx_subchunk.intersects(idx_red):
        raise ValueError(
            "`idx_red` must intersect `idx_subchunk`;"
            " `idx_red`: {idx_red}, `idx_subchunk`: {idx_subchunk}"
        )
    x_pad, y_pad, z_pad = processing_blend_pad
    weight = get_weight_template(
        processing_blend_mode,
        tuple(idx_subchunk.shape),
        x_pad,
        y_pad,
        z_pad,
        *idx_subchunk.aligned(idx_roi),
    )

    intscn = idx_red.intersection(idx_subchunk)
    _, intscn_in_subchunk = intscn.get_intersection_and_subindex(idx_subchunk)

    return weight[intscn_in_subchunk].unsqueeze(0)


def set_allow_cache(*args, **kwargs):
    newargs = []
    newkwargs = {}
    for arg in args:
        if (
            isinstance(arg, VolumetricBasedLayerProtocol)
            and not arg.backend.is_local
            and not arg.backend.allow_cache
        ):
            newarg = arg.with_changes(backend=arg.backend.with_changes(allow_cache=True))
        else:
            newarg = arg
        newargs.append(newarg)

    for k, v in kwargs.items():
        if (
            isinstance(v, VolumetricBasedLayerProtocol)
            and not v.backend.is_local
            and not v.backend.allow_cache
        ):
            newv = v.with_changes(backend=v.backend.with_changes(allow_cache=True))
        else:
            newv = v
        newkwargs[k] = newv

    return newargs, newkwargs


def clear_cache(*args, **kwargs):
    for arg in args:
        if isinstance(arg, VolumetricBasedLayerProtocol):
            if not arg.backend.is_local and arg.backend.allow_cache:
                arg.backend.clear_cache()
    for kwarg in kwargs.values():
        if isinstance(kwarg, VolumetricBasedLayerProtocol):
            if not kwarg.backend.is_local and kwarg.backend.allow_cache:
                kwarg.backend.clear_cache()


def delete_if_local(*args, **kwargs):
    filesystem = fsspec.filesystem("file")
    for arg in args:
        if isinstance(arg, VolumetricBasedLayerProtocol):
            if arg.backend.is_local:
                filesystem.delete(arg.backend.name, recursive=True)
    for kwarg in kwargs.values():
        if isinstance(kwarg, VolumetricBasedLayerProtocol):
            if kwarg.backend.is_local:
                filesystem.delete(kwarg.backend.name, recursive=True)


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
    processing_blend_mode: Literal["linear", "quadratic", "max", "defer"] = "quadratic"
    processing_gap: Optional[Vec3D[int]] = None
    intermediaries_dir: Optional[str] = None
    allow_cache: bool = False
    clear_cache_on_return: bool = False
    force_intermediaries: bool = False
    use_checkerboarding: bool = attrs.field(init=False)
    processing_chunker: VolumetricIndexChunker = attrs.field(init=False)
    flow_id: str = "no_id"
    l0_chunks_per_task: int = 0
    op_worker_type: str | None = None
    reduction_worker_type: str | None = None

    @property
    def _intermediaries_are_local(self) -> bool:
        assert self.intermediaries_dir is not None
        return self.intermediaries_dir.startswith("file://") or "//" not in self.intermediaries_dir

    def _get_backend_chunk_size_to_use(self, dst) -> Vec3D[int]:
        assert self.processing_blend_pad is not None
        backend_chunk_size = deepcopy(self.processing_chunk_size)
        dst_backend_chunk_size = dst.backend.get_chunk_size(self.dst_resolution)

        for i in range(3):
            if backend_chunk_size[i] == 1 or self.processing_blend_pad[i] == 0:
                continue
            if self.processing_chunk_size[i] % 2 != 0:
                raise ValueError(
                    "`processing_chunk_size` must be divisible by 2 at least once in"
                    " blended dimensions that are not 1;"
                    " received {self.processing_chunk_size[i]} against {dst_backend_chunk_size[i]}"
                )
            # The following line is the non-mutable version of backend_chunk_size[i] //= 2
            backend_chunk_size = Vec3D[int](
                *itertools.chain(
                    backend_chunk_size[0:i],
                    (backend_chunk_size[i] // 2,),
                    backend_chunk_size[i + 1 :],
                )
            )
            while backend_chunk_size[i] > dst_backend_chunk_size[i]:
                if backend_chunk_size[i] % 2 != 0:
                    raise ValueError(
                        "`processing_chunk_size` must be divisible by 2 continuously until"
                        " it is smaller than the `dst` backend's chunk size, or be 1, in"
                        f" dims that are blended; received {self.processing_chunk_size[i]}"
                    )
                backend_chunk_size = Vec3D[int](
                    *itertools.chain(
                        backend_chunk_size[0:i],
                        (backend_chunk_size[i] // 2,),
                        backend_chunk_size[i + 1 :],
                    )
                )

        return backend_chunk_size

    def __attrs_post_init__(self):  # pylint: disable=too-many-branches
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
        if self.processing_gap is None:
            self.processing_gap = Vec3D[int](0, 0, 0)

        if self.use_checkerboarding:
            if not self.processing_blend_pad <= self.processing_chunk_size // 2:
                raise ValueError(
                    f" `processing_blend_pad` must be less than or equal to"
                    f" half of `processing_chunk_size`; received {self.processing_blend_pad}",
                    f" which is larger than {self.processing_chunk_size // 2}",
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
            chunk_size=self.processing_chunk_size,
            resolution=self.dst_resolution,
            stride=self.processing_chunk_size + self.processing_gap,
        )

        if self.max_reduction_chunk_size is None:
            self.max_reduction_chunk_size_final = self.processing_chunk_size
        else:
            self.max_reduction_chunk_size_final = self.max_reduction_chunk_size

    def _get_temp_dst(
        self,
        dst: VolumetricBasedLayerProtocol,
        idx: VolumetricIndex,
        prefix: Optional[Any] = None,
        suffix: Optional[Any] = None,
    ) -> VolumetricBasedLayerProtocol:
        assert self.intermediaries_dir is not None
        temp_name = f"{prefix}_{self.op.__class__.__name__}_temp_{idx.pformat()}_{suffix}"
        allow_cache = self.allow_cache and not self._intermediaries_are_local
        if self.use_checkerboarding:
            backend_chunk_size_to_use = self._get_backend_chunk_size_to_use(dst)
        else:
            backend_chunk_size_to_use = self.processing_chunk_size
        backend_temp_base = dst.backend
        backend_temp = backend_temp_base.with_changes(
            name=path.join(self.intermediaries_dir, temp_name),
            voxel_offset_res=(idx.start - backend_chunk_size_to_use, self.dst_resolution),
            chunk_size_res=(backend_chunk_size_to_use, self.dst_resolution),
            dataset_size_res=(
                dst.backend.get_dataset_size(self.dst_resolution) + 2 * backend_chunk_size_to_use,
                self.dst_resolution,
            ),
            enforce_chunk_aligned_writes=False,
            allow_cache=allow_cache,
            use_compression=False,
        )
        return dst.with_procs(read_procs=()).with_changes(backend=backend_temp)

    def _make_task(
        self,
        arg: Tuple[
            VolumetricIndex, VolumetricBasedLayerProtocol | None, dict[str, Any]
        ],  # cannot type with P.kwargs
    ) -> mazepa.tasks.Task[R_co]:
        return self.op.make_task(idx=arg[0], dst=arg[1], **arg[2]).with_worker_type(
            self.op_worker_type
        )

    def make_tasks_without_checkerboarding(
        self,
        idx_chunks: List[VolumetricIndex],
        dst: VolumetricBasedLayerProtocol | None,
        op_kwargs: P.kwargs,
    ) -> List[mazepa.tasks.Task[R_co]]:
        if len(idx_chunks) > MULTIPROCESSING_NUM_TASKS_THRESHOLD:
            with multiprocessing.get_context('fork').Pool() as pool_obj:
                tasks = pool_obj.map(
                    self._make_task,
                    zip(idx_chunks, itertools.repeat(dst), itertools.repeat(op_kwargs)),
                )
        else:
            tasks = list(
                map(
                    self._make_task,
                    zip(idx_chunks, itertools.repeat(dst), itertools.repeat(op_kwargs)),
                )
            )
        return tasks

    def make_tasks_with_intermediaries(  # pylint: disable=too-many-locals
        self,
        idx: VolumetricIndex,
        dst: VolumetricBasedLayerProtocol,
        op_kwargs: P.kwargs,
    ) -> Tuple[List[mazepa.tasks.Task[R_co]], VolumetricBasedLayerProtocol | None]:
        dst_temp = self._get_temp_dst(dst, idx, self.flow_id)
        have_processing_gap = self.processing_gap is not None and self.processing_gap != Vec3D[
            int
        ](0, 0, 0)
        # TODO: remove "expand"; see https://github.com/ZettaAI/zetta_utils/issues/648
        idx_chunks = self.processing_chunker(
            idx,
            mode="expand" if have_processing_gap else "exact",
            chunk_id_increment=self.l0_chunks_per_task,
        )
        tasks = self.make_tasks_without_checkerboarding(idx_chunks, dst_temp, op_kwargs)
        return tasks, dst_temp

    def make_tasks_with_checkerboarding(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        self,
        idx: VolumetricIndex,
        red_chunks: List[VolumetricIndex],
        red_shape: Vec3D[int],
        dst: VolumetricBasedLayerProtocol,
        op_kwargs: P.kwargs,
    ) -> Tuple[
        List[mazepa.tasks.Task[R_co]],
        List[List[VolumetricIndex]],
        List[List[VolumetricBasedLayerProtocol]],
        List[VolumetricBasedLayerProtocol],
    ]:
        assert self.intermediaries_dir is not None
        assert self.processing_blend_pad is not None
        """
        Makes tasks that can be reduced to the final output, and for each reduction chunk,
        the list of VolumetricIndices and the VolumetricBasedLayerProtocols that can be
        reduced to the final output, as well as the temporary destination layers.
        """
        tasks: List[mazepa.tasks.Task[R_co]] = []
        red_chunks_task_idxs: List[List[VolumetricIndex]] = [[] for _ in red_chunks]
        red_chunks_temps: List[List[VolumetricBasedLayerProtocol]] = [[] for _ in red_chunks]
        red_chunks_3d = np.array(red_chunks, dtype=object).reshape(red_shape, order="F")
        dst_temps: List[VolumetricBasedLayerProtocol] = []

        next_chunk_id = idx.chunk_id
        for chunker, chunker_idx in self.processing_chunker.split_into_nonoverlapping_chunkers(
            self.processing_blend_pad
        ):
            """
            Prepare the temporary destination by allowing non-aligned writes, aligning the voxel
            offset of the temporary destination with where the given idx starts, and setting the
            backend chunk size to half of the processing chunk size. Furthermore, skip caching
            if the temporary destination happens to be local.
            """
            dst_temp = self._get_temp_dst(
                dst, idx, self.flow_id, "_".join(str(i) for i in chunker_idx)
            )
            dst_temps.append(dst_temp)
            with suppress_type_checks():
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
                idx_expanded.chunk_id = next_chunk_id

                task_shape = chunker.get_shape(
                    idx_expanded,
                    stride_start_offset=idx_expanded.start,
                    mode="shrink",
                )
                task_idxs = np.array(
                    chunker(
                        idx_expanded,
                        stride_start_offset=idx_expanded.start,
                        mode="shrink",
                        chunk_id_increment=self.l0_chunks_per_task,
                    ),
                    dtype=object,
                ).reshape(task_shape, order="F")

                if task_shape[0] * task_shape[1] * task_shape[2] == 0:
                    continue

                next_chunk_id = task_idxs[-1, -1, -1].chunk_id + self.l0_chunks_per_task

                if len(task_idxs) > MULTIPROCESSING_NUM_TASKS_THRESHOLD:
                    with multiprocessing.get_context('fork').Pool() as pool_obj:
                        tasks_split = pool_obj.map(
                            self._make_task,
                            zip(
                                task_idxs.ravel(),
                                itertools.repeat(dst_temp),
                                itertools.repeat(op_kwargs),
                            ),
                        )
                else:
                    tasks_split = list(
                        map(
                            self._make_task,
                            zip(
                                task_idxs.ravel(),
                                itertools.repeat(dst_temp),
                                itertools.repeat(op_kwargs),
                            ),
                        )
                    )
                tasks += tasks_split

                red_stops = [
                    [chunk.stop[0] for chunk in red_chunks_3d[:, 0, 0]],
                    [chunk.stop[1] for chunk in red_chunks_3d[0, :, 0]],
                    [chunk.stop[2] for chunk in red_chunks_3d[0, 0, :]],
                ]

                task_starts = [
                    [task_idxs[i, 0, 0].start[0] for i in range(task_shape[0])],
                    [task_idxs[0, i, 0].start[1] for i in range(task_shape[1])],
                    [task_idxs[0, 0, i].start[2] for i in range(task_shape[2])],
                ]

                task_to_red_chunks: list[dict[int, int]] = [{}, {}, {}]

                for axis in range(3):
                    red_stop_ind = 0
                    for i, task_start in enumerate(task_starts[axis]):
                        try:
                            while not task_start < red_stops[axis][red_stop_ind]:
                                red_stop_ind += 1
                        # This case catches the case where the chunk is entirely outside
                        # any reduction chunk; this can happen if, for instance,
                        # roi_crop_pad is set to [0, 0, 1] and the processing_chunk_size
                        # is [X, X, 1].
                        except IndexError as e:
                            raise ValueError(
                                f"The processing chunk starting at `{task_start}` in axis {axis}"
                                " does not correspond to any reduction chunk; please check the "
                                "`roi_crop_pad` and the `processing_chunk_size`."
                            ) from e
                        task_to_red_chunks[axis][i] = red_stop_ind

                flat_red_ind_offsets = set(
                    i + j * red_shape[0] + k * red_shape[0] * red_shape[1]
                    for i, j, k in itertools.product(range(3), repeat=3)
                )
                for i, task_ind in enumerate(np.ndindex(task_idxs.shape)):
                    task_idx = task_idxs[*task_ind]
                    red_ind = Vec3D(
                        *(task_to_red_chunks[axis][task_ind[axis]] for axis in range(3))
                    )
                    flat_red_ind = (
                        red_ind[0]
                        + red_shape[0] * red_ind[1]
                        + red_shape[0] * red_shape[1] * red_ind[2]
                    )
                    if task_idx.contained_in(red_chunks[flat_red_ind]):
                        red_chunks_task_idxs[flat_red_ind].append(task_idx)
                        red_chunks_temps[flat_red_ind].append(dst_temp)
                    else:
                        flat_red_inds = [offset + flat_red_ind for offset in flat_red_ind_offsets]
                        for i in flat_red_inds:
                            if i < len(red_chunks) and task_idx.intersects(red_chunks[i]):
                                red_chunks_task_idxs[i].append(task_idx)
                                red_chunks_temps[i].append(dst_temp)
        return (tasks, red_chunks_task_idxs, red_chunks_temps, dst_temps)

    def flow(  # pylint:disable=too-many-branches, too-many-statements
        self,
        idx: VolumetricIndex,
        dst: VolumetricBasedLayerProtocol | None,
        op_args: P.args,
        op_kwargs: P.kwargs,
    ) -> mazepa.FlowFnReturnType:
        assert len(op_args) == 0
        assert self.roi_crop_pad is not None
        assert self.processing_blend_pad is not None
        # set caching for all VolumetricBasedLayerProtocols as desired
        if self.allow_cache:
            op_args, op_kwargs = set_allow_cache(*op_args, **op_kwargs)

        logger.debug(f"Breaking {idx} into chunks with {self.processing_chunker}.")

        # cases without checkerboarding
        if not self.use_checkerboarding and not self.force_intermediaries:
            idx_chunks = self.processing_chunker(
                idx, mode="exact", chunk_id_increment=self.l0_chunks_per_task
            )
            tasks = self.make_tasks_without_checkerboarding(idx_chunks, dst, op_kwargs)
            logger.info(f"Submitting {len(tasks)} processing tasks from operation {self.op}.")
            yield tasks
        elif not self.use_checkerboarding and self.force_intermediaries:
            assert dst is not None
            tasks, dst_temp = self.make_tasks_with_intermediaries(idx, dst, op_kwargs)
            logger.info(f"Submitting {len(tasks)} processing tasks from operation {self.op}.")
            yield tasks
            yield mazepa.Dependency()
            if self.processing_gap is None:
                self.processing_gap = Vec3D[int](0, 0, 0)
            if self.processing_gap != Vec3D[int](0, 0, 0):
                copy_chunk_size = (
                    dst.backend.get_chunk_size(self.dst_resolution) - self.processing_gap // 2
                )
            elif not self.max_reduction_chunk_size_final >= dst.backend.get_chunk_size(
                self.dst_resolution
            ):
                copy_chunk_size = dst.backend.get_chunk_size(self.dst_resolution)
            else:
                copy_chunk_size = self.max_reduction_chunk_size_final

            reduction_chunker = VolumetricIndexChunker(
                chunk_size=dst.backend.get_chunk_size(self.dst_resolution)
                - self.processing_gap // 2,
                resolution=self.dst_resolution,
                max_superchunk_size=copy_chunk_size,
                offset=-self.processing_gap // 2,
            )
            logger.debug(
                f"Breaking {idx} into chunks to be copied from the intermediary layer"
                f" with {reduction_chunker}."
            )
            stride_start_offset = dst.backend.get_voxel_offset(self.dst_resolution)
            red_chunks = reduction_chunker(
                idx, mode="exact", stride_start_offset=stride_start_offset
            )
            tasks_reduce = [
                Copy()
                .make_task(
                    src=dst_temp, dst=dst.with_procs(read_procs=(), write_procs=()), idx=red_chunk
                )
                .with_worker_type(self.reduction_worker_type)
                for red_chunk in red_chunks
            ]
            logger.info(
                "Copying temporary destination backend into the final destination:"
                f" Submitting {len(tasks_reduce)} tasks."
            )
            yield tasks_reduce
            yield mazepa.Dependency()
            clear_cache(dst_temp)
            delete_if_local(dst_temp)
        # cases with checkerboarding
        elif self.processing_blend_mode == "defer":
            assert dst is not None
            stride_start_offset = dst.backend.get_voxel_offset(self.dst_resolution)
            (tasks, _, _, dst_temps,) = self.make_tasks_with_checkerboarding(
                idx.padded(self.roi_crop_pad), [idx], Vec3D(1, 1, 1), dst, op_kwargs
            )
            logger.info(
                "Writing to intermediate destinations:\n"
                f" Submitting {len(tasks)} processing tasks from operation {self.op}.\n"
                f"Note that because blending is deferred, {dst.pformat()} will NOT "
                f"contain the final output."
            )
            yield tasks
            yield mazepa.Dependency()
        else:
            assert dst is not None
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
            logger.debug(
                f"Breaking {idx} into reduction chunks with checkerboarding"
                f" with {reduction_chunker}. Processing chunks will use the padded index"
                f" {idx.padded(self.roi_crop_pad)} and be chunked with {self.processing_chunker}."
            )
            stride_start_offset = dst.backend.get_voxel_offset(self.dst_resolution)
            red_chunks = reduction_chunker(
                idx, mode="exact", stride_start_offset=stride_start_offset
            )
            red_shape = reduction_chunker.get_shape(
                idx, mode="exact", stride_start_offset=stride_start_offset
            )
            (
                tasks,
                red_chunks_task_idxs,
                red_chunks_temps,
                dst_temps,
            ) = self.make_tasks_with_checkerboarding(
                idx.padded(self.roi_crop_pad), red_chunks, red_shape, dst, op_kwargs
            )
            logger.info(
                "Writing to temporary destinations:\n"
                f" Submitting {len(tasks)} processing tasks from operation {self.op}."
            )
            yield tasks
            yield mazepa.Dependency()
            reducer: ReduceOperation
            if self.processing_blend_mode == "max":
                reducer = ReduceNaive()
            else:
                reducer = ReduceByWeightedSum(self.processing_blend_mode)
            tasks_reduce = [
                reducer.make_task(
                    src_idxs=red_chunk_task_idxs,
                    src_layers=red_chunk_temps,
                    red_idx=red_chunk,
                    roi_idx=idx.padded(self.roi_crop_pad + self.processing_blend_pad),
                    dst=dst.with_procs(read_procs=(), write_procs=()),
                    processing_blend_pad=self.processing_blend_pad,
                ).with_worker_type(self.reduction_worker_type)
                for (
                    red_chunk_task_idxs,
                    red_chunk_temps,
                    red_chunk,
                ) in zip(red_chunks_task_idxs, red_chunks_temps, red_chunks)
            ]
            logger.info(
                "Collating temporary destination backends into the final destination:"
                f" Submitting {len(tasks_reduce)} tasks."
            )
            yield tasks_reduce
            yield mazepa.Dependency()
            clear_cache(*dst_temps)
            delete_if_local(*dst_temps)
        if self.clear_cache_on_return:
            clear_cache(*op_args, **op_kwargs)

        return tasks

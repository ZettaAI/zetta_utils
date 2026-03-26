from __future__ import annotations

import time
from abc import ABC
from contextlib import nullcontext
from typing import (
    List,
    Literal,
    Tuple,
    assert_never,
)

import attrs
import cachetools
import numpy as np
import torch
from typeguard import suppress_type_checks

from zetta_utils import log, mazepa
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import (
    VolumetricBasedLayerProtocol,
    VolumetricIndex,
)
from zetta_utils.mazepa import semaphore
from zetta_utils.tensor_ops import convert

_weights_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=16)

logger = log.get_logger("zetta_utils")


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
            reduce_start = time.time()
            res = np.zeros(
                (dst.backend.num_channels, *red_idx.shape),
                dtype=dst.backend.dtype,
            )
            assert len(src_layers) > 0
            if processing_blend_pad != Vec3D[int](0, 0, 0):
                for src_idx, layer in zip(src_idxs, src_layers):
                    intscn, subidx = src_idx.get_intersection_and_subindex(red_idx)
                    subidx_channels = (slice(0, res.shape[0]), *subidx)
                    read_ctx = nullcontext() if layer.backend.is_local else semaphore("read")
                    with read_ctx:
                        res[subidx_channels] = np.maximum(res[subidx_channels], layer[intscn])
            else:
                for src_idx, layer in zip(src_idxs, src_layers):
                    intscn, subidx = src_idx.get_intersection_and_subindex(red_idx)
                    subidx_channels = (slice(0, res.shape[0]), *subidx)
                    read_ctx = nullcontext() if layer.backend.is_local else semaphore("read")
                    with read_ctx:
                        res[subidx_channels] = layer[intscn]
            reduce_time = time.time() - reduce_start
            if np.any(res):
                write_start = time.time()
                write_ctx = nullcontext() if dst.backend.is_local else semaphore("write")
                with write_ctx:
                    dst[red_idx] = res
                write_time = time.time() - write_start
            else:
                write_time = 0.0
            logger.info(
                f"ReduceNaive: {len(src_layers)} sources, "
                f"reduce: {reduce_time:.3f}s, write: {write_time:.3f}s"
            )


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
            reduce_start = time.time()
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
                    read_ctx = nullcontext() if layer.backend.is_local else semaphore("read")
                    with read_ctx:
                        if not is_floating_point_dtype(dst.backend.dtype):
                            # Temporarily convert integer cutout to float for rounding
                            res[tuple(subidx_channels)] = (
                                res[tuple(subidx_channels)]
                                + layer[intscn].astype(float) * weight.numpy()
                            )
                        else:
                            res[tuple(subidx_channels)] = (
                                res[tuple(subidx_channels)] + layer[intscn] * weight.numpy()
                            )

                if not is_floating_point_dtype(dst.backend.dtype):
                    res = res.round().to(dtype=convert.to_torch_dtype(dst.backend.dtype))
            else:
                for src_idx, layer in zip(src_idxs, src_layers):
                    intscn, subidx = src_idx.get_intersection_and_subindex(red_idx)
                    subidx_channels = [slice(0, res.shape[0])] + list(subidx)
                    read_ctx = nullcontext() if layer.backend.is_local else semaphore("read")
                    with read_ctx:
                        res.numpy()[tuple(subidx_channels)] = layer[intscn]
            reduce_time = time.time() - reduce_start
            if res.any():
                write_start = time.time()
                write_ctx = nullcontext() if dst.backend.is_local else semaphore("write")
                with write_ctx:
                    dst[red_idx] = res
                write_time = time.time() - write_start
            else:
                write_time = 0.0
            logger.info(
                f"ReduceByWeightedSum: {len(src_layers)} sources, "
                f"reduce: {reduce_time:.3f}s, write: {write_time:.3f}s"
            )


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
    else:
        assert_never(processing_blend_mode)

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

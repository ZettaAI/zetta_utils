# pylint: disable=missing-docstring
from __future__ import annotations

import ast
from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Union

import attrs
import cachetools
import numpy as np
import tensorstore
import torch
from cachetools.keys import hashkey

from zetta_utils import tensor_ops
from zetta_utils.geometry import Vec3D

from .. import VolumetricBackend, VolumetricIndex
from ..cloudvol import (
    InfoExistsModes,
    PrecomputedInfoSpec,
    _get_info,
    _info_cache,
    _info_hash_key,
)

_ts_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=500)
_ts_hash_key = hashkey

# str for hashability
@cachetools.cached(_ts_cache, key=_ts_hash_key)
def _get_ts_at_resolution(path, resolution: Optional[str] = None) -> tensorstore.TensorStore:
    spec: Dict[str, Any] = {
        "driver": "neuroglancer_precomputed",
        "kvstore": path,
    }
    if resolution is not None:
        spec["scale_metadata"] = {"resolution": ast.literal_eval(resolution)}
    result = tensorstore.open(spec).result()
    return result


@attrs.mutable
class TSBackend(VolumetricBackend):  # pylint: disable=too-few-public-methods
    """
    Backend for peforming IO on Neuroglancer datasts using TensorStore library.
    Read data will be a ``torch.Tensor`` in ``BCXYZ`` dimension order.
    Write data is expected to be a ``torch.Tensor`` or ``np.ndarray`` in ``BCXYZ``
    dimension order.
    :param path: Precomputed path.
    :param info_spec: Specification for the info file for the layer. If None, the
        info is assumed to exist.
    :param on_info_exists: Behavior mode for when both `info_spec` is given and
        the layer info already exists.

    """

    path: str
    info_spec: Optional[PrecomputedInfoSpec] = None
    on_info_exists: InfoExistsModes = "expect_same"

    def __attrs_post_init__(self):
        if self.info_spec is not None:
            new_info = self.info_spec.make_info()
            if new_info is not None:
                try:
                    existing_info = _get_info(self.path)
                except FileNotFoundError:
                    existing_info = None

                if (
                    existing_info is not None
                    and self.on_info_exists == "expect_same"
                    and new_info != existing_info
                ):
                    raise RuntimeError(
                        f"Info created by the info_spec {self.info_spec} is not equal to "
                        f"info existing at '{self.path}' "
                        "while `on_info_exists` is set to 'expect_same'"
                    )
                if existing_info != new_info:
                    self.info_spec.write_info(self.path)
            self.info_spec.reference_path = self.path
            _info_cache[_info_hash_key(self.path)] = new_info
            _ts_cache.clear()
        else:
            _get_info(self.path)

    @property
    def name(self) -> str:  # pragma: no cover
        return self.path

    @name.setter
    def name(self, name: str) -> None:  # pragma: no cover
        raise NotImplementedError(
            "cannot set `name` for CVBackend directly;"
            " use `backend.with_changes(name='name')` instead."
        )

    @property
    def dtype(self) -> torch.dtype:
        try:
            result = _get_ts_at_resolution(self.path)
            dtype = result.dtype.name
            return getattr(torch, dtype)
        except Exception as e:
            raise e

    @property
    # TODO: Figure out a way to access 'multiscale metadata' directly
    def num_channels(self) -> int:  # pragma: no cover
        result = _get_ts_at_resolution(self.path)
        return result.shape[-1]

    @property
    def is_local(self) -> bool:  # pragma: no cover
        return self.path.startswith("file://")

    @property
    def enforce_chunk_aligned_writes(self) -> bool:  # pragma: no cover
        return False

    @enforce_chunk_aligned_writes.setter
    def enforce_chunk_aligned_writes(self, value: bool) -> None:  # pragma: no cover
        raise NotImplementedError(
            "cannot set `enforce_chunk_aligned_writes` for TSBackend; can only be set to `False`"
        )

    @property
    def allow_cache(self) -> bool:  # pragma: no cover
        return False

    @allow_cache.setter
    def allow_cache(self, value: Union[bool, str]) -> None:  # pragma: no cover
        raise NotImplementedError(
            "cannot set `allow_cache` for CVBackend directly;"
            " use `backend.with_changes(allow_cache=value:Union[bool, str])` instead."
        )

    @property
    def use_compression(self) -> bool:  # pragma: no cover
        return False

    @use_compression.setter
    def use_compression(self, value: bool) -> None:  # pragma: no cover
        raise NotImplementedError(
            "cannot set `use_compression` for TSBackend; can only be set to `False`"
        )

    # TODO: Implement context-based caching
    def clear_cache(self) -> None:  # pragma: no cover
        pass

    def read(self, idx: VolumetricIndex) -> torch.Tensor:
        # Data out: bcxyz
        ts = _get_ts_at_resolution(self.path, str(list(idx.resolution)))
        data_raw = np.array(ts[idx.to_slices()])
        result_np = np.transpose(data_raw, (3, 0, 1, 2))
        result = tensor_ops.to_torch(result_np)
        return result

    def write(self, idx: VolumetricIndex, data: torch.Tensor):
        # Data in: bcxyz
        # Write format: xyzc (b == 1)
        data_np = tensor_ops.convert.to_np(data)
        if data_np.size == 1 and len(data_np.shape) == 1:
            data_final = data_np[0]
        elif len(data_np.shape) == 4:
            data_final = np.transpose(data_np, (1, 2, 3, 0))
        else:
            raise ValueError(
                "Data written to CloudVolume backend must be in `cxyz` dimension format, "
                f"but got a tensor of with ndim == {data_np.ndim}"
            )

        ts = _get_ts_at_resolution(self.path, str(list(idx.resolution)))
        slices = idx.to_slices()
        ts[slices] = data_final

    def as_type(self, backend_type) -> VolumetricBackend:  # pragma: no cover # type: ignore
        type_args = [f.name for f in attrs.fields(backend_type)]
        keys_to_use = set(dir(self)).intersection(type_args)
        kwargs_to_use = {k: getattr(self, k) for k in keys_to_use}
        return backend_type(**kwargs_to_use)

    def with_changes(self, **kwargs) -> TSBackend:
        """Currently untyped. Supports:
        "name" = value: str
        "allow_cache" = value: Union[bool, str] - must be False for TensorStoreBackend
        "enforce_chunk_aligned_writes" = value: bool - must be False for TensorStoreBackend
        "voxel_offset_res" = (voxel_offset, resolution): Tuple[Vec3D[int], Vec3D]
        "chunk_size_res" = (chunk_size, resolution): Tuple[Vec3D[int], Vec3D]
        """
        assert self.info_spec is not None

        info_spec = deepcopy(self.info_spec)

        implemented_keys = [
            "name",
            "allow_cache",
            "enforce_chunk_aligned_writes",
            "voxel_offset_res",
            "chunk_size_res",
        ]
        keys_to_kwargs = {"name": "path"}
        keys_to_infospec_fn = {
            "voxel_offset_res": info_spec.set_voxel_offset,
            "chunk_size_res": info_spec.set_chunk_size,
        }
        keys_to_assert = {"allow_cache": False, "enforce_chunk_aligned_writes": False}
        evolve_kwargs = {}
        for k, v in kwargs.items():
            if k not in implemented_keys:
                raise KeyError(f"key `{k}` received, expected one of `{implemented_keys}`")
            if k in keys_to_kwargs:
                evolve_kwargs[keys_to_kwargs[k]] = v
            if k in keys_to_infospec_fn:
                keys_to_infospec_fn[k](v)
            if k in keys_to_assert:
                if v != keys_to_assert[k]:
                    raise ValueError(
                        f"key `{k}` received with value `{v}`, but is required to be "
                        f"`{keys_to_assert[k]}`"
                    )

        return attrs.evolve(
            self,
            **evolve_kwargs,
            info_spec=info_spec,
            on_info_exists="overwrite",
        )

    def get_voxel_offset(self, resolution: Vec3D) -> Vec3D[int]:
        ts = _get_ts_at_resolution(self.path, str(list(resolution)))
        return Vec3D[int](*ts.chunk_layout.grid_origin[0:3])

    def get_chunk_size(self, resolution: Vec3D) -> Vec3D[int]:
        ts = _get_ts_at_resolution(self.path, str(list(resolution)))
        return Vec3D[int](*ts.chunk_layout.read_chunk.shape[0:3])

    def get_chunk_aligned_index(  # pragma: no cover
        self, idx: VolumetricIndex, mode: Literal["expand", "shrink", "round"]
    ) -> VolumetricIndex:
        offset = self.get_voxel_offset(idx.resolution) * idx.resolution
        chunk_size = self.get_chunk_size(idx.resolution) * idx.resolution

        if mode == "expand":
            bbox_aligned = idx.bbox.snapped(offset, chunk_size, "expand")
        elif mode == "shrink":
            bbox_aligned = idx.bbox.snapped(offset, chunk_size, "shrink")
        elif mode == "round":  # pragma: no cover
            raise NotImplementedError("'round' mode not supported for TensorStore backends")
        else:
            raise NotImplementedError(
                f"mode must be set to 'expand', 'shrink', or 'round'; received '{mode}'"
            )
        return VolumetricIndex(resolution=idx.resolution, bbox=bbox_aligned)

    def assert_idx_is_chunk_aligned(self, idx: VolumetricIndex) -> None:  # pragma: no cover
        """check that the idx given is chunk_aligned, and give suggestions"""
        idx_expanded = self.get_chunk_aligned_index(idx, mode="expand")
        idx_shrunk = self.get_chunk_aligned_index(idx, mode="shrink")

        if idx != idx_expanded:
            raise ValueError(
                "The specified BBox3D is not chunk-aligned with the VolumetricLayer at"
                + f" `{self.name}`;\nin {tuple(idx.resolution)} {idx.bbox.unit} voxels:"
                + f" offset: {self.get_voxel_offset(idx.resolution)},"
                + f" chunk_size: {self.get_chunk_size(idx.resolution)}\n"
                + f"Received BBox3D: {idx.pformat()}\n"
                + "Nearest chunk-aligned BBox3Ds:\n"
                + f" - expanded : {idx_expanded.pformat()}\n"
                + f" - shrunk   : {idx_shrunk.pformat()}"
            )

    def pformat(self) -> str:  # pragma: no cover
        return self.name

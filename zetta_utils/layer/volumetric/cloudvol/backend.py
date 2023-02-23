# pylint: disable=missing-docstring
from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Tuple, Union

import attrs
import cachetools
import cloudvolume as cv
import fsspec
import fsspec.asyn
import numpy as np
import torch
from cachetools.keys import hashkey
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox
from typeguard import typechecked

from zetta_utils import builder, tensor_ops
from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D

from .. import VolumetricBackend, VolumetricIndex

_info_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=500)
_info_hash_key = hashkey


@cachetools.cached(_info_cache, key=_info_hash_key)
def _get_info(path: str) -> Dict[str, Any]:
    if not path.endswith("/info"):
        path = os.path.join(path, "info")
    try:
        fsspec.asyn.reset_lock()  # https://github.com/fsspec/gcsfs/issues/379
        with fsspec.open(path) as f:
            result = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"'{path}' does not have an infofile.") from e
    return result


# To avoid reloading info file - note that an empty provenance is passed
# since otherwise the CloudVolume's __new__ will download the provenance
def get_cv_cached(cloudpath, *args, **kwargs):
    return CloudVolume(cloudpath, info=_get_info(cloudpath), provenance={}, *args, **kwargs)


@builder.register("PrecomputedInfoSpec")
@typechecked
@attrs.mutable
class PrecomputedInfoSpec:
    reference_path: Optional[str] = None
    field_overrides: Optional[Dict[str, Any]] = None
    default_chunk_size: Optional[IntVec3D] = None
    default_voxel_offset: Optional[IntVec3D] = None
    chunk_size_map: Optional[Dict[str, IntVec3D]] = None
    voxel_offset_map: Optional[Dict[str, IntVec3D]] = None
    data_type: Optional[str] = None
    # ensure_scales: Optional[Iterable[int]] = None

    def set_voxel_offset(self, voxel_offset_and_res: Tuple[IntVec3D, Vec3D]) -> None:
        voxel_offset, resolution = voxel_offset_and_res
        key = "_".join([_str(v) for v in resolution])
        if self.voxel_offset_map is None:
            self.voxel_offset_map = {}

        self.voxel_offset_map[key] = voxel_offset

    def set_chunk_size(self, chunk_size_and_res: Tuple[IntVec3D, Vec3D]) -> None:
        chunk_size, resolution = chunk_size_and_res
        key = "_".join([_str(v) for v in resolution])
        if self.chunk_size_map is None:
            self.chunk_size_map = {}
        self.chunk_size_map[key] = chunk_size

    def make_info(  # pylint: disable=too-many-branches, consider-iterating-dictionary
        self,
    ) -> Optional[Dict[str, Any]]:
        if self.reference_path is None and self.field_overrides is None:
            result = None
        else:
            field_overrides = self.field_overrides
            if field_overrides is None:
                field_overrides = {}
            reference_info = {}  # type: Dict[str, Any]
            if self.reference_path is not None:
                reference_info = _get_info(self.reference_path)
            result = deepcopy({**reference_info, **field_overrides})
            if self.default_chunk_size is not None:
                for e in result["scales"]:
                    e["chunk_sizes"] = [[*self.default_chunk_size]]
            if self.chunk_size_map is not None:
                for e in result["scales"]:
                    if e["key"] in self.chunk_size_map.keys():
                        e["chunk_sizes"] = [[*self.chunk_size_map[e["key"]]]]
            if self.default_voxel_offset is not None:
                for e in result["scales"]:
                    e["voxel_offset"] = [*self.default_voxel_offset]
            if self.voxel_offset_map is not None:
                for e in result["scales"]:
                    if e["key"] in self.voxel_offset_map.keys():
                        e["voxel_offset"] = [*self.voxel_offset_map[e["key"]]]
            if self.data_type is not None:
                result["data_type"] = self.data_type

            # if self.ensure_scales is not None:  # pragma: no cover
            #    raise NotImplementedError()

        return result


InfoExistsModes = Literal["expect_same", "overwrite"]


def _str(n: float) -> str:  # pragma: no cover
    if int(n) == n:
        return str(int(n))
    return str(n)


@builder.register("CVBackend")
# @typechecked raises stop iteration, TOOD: make an issue
@attrs.mutable
class CVBackend(VolumetricBackend):  # pylint: disable=too-few-public-methods
    """
    Backend for peforming IO on Neuroglancer datasts using CloudVolume library.
    Read data will be a ``torch.Tensor`` in ``BCXYZ`` dimension order.
    Write data is expected to be a ``torch.Tensor`` or ``np.ndarray`` in ``BCXYZ``
    dimension order.
    :param path: CloudVolume path.
    :param cv_kwargs: Parameters that will be passed to the CloudVolume constructor.
                    print(new_info == _get_info(self.path))
        ``mip`` keyword must not be present in ``cv_kwargs``, as the read resolution
        is passed in as a part of index to the backend.
    :param info_spec: Specification for the info file for the layer. If None, the
        info is assumed to exist.
    :param on_info_exists: Behavior mode for when both `info_spec` is given and
        the layer info already exists.

    """

    path: str
    cv_kwargs: Dict[str, Any] = attrs.field(factory=dict)
    info_spec: Optional[PrecomputedInfoSpec] = None
    on_info_exists: InfoExistsModes = "expect_same"

    def __attrs_post_init__(self):
        if "mip" in self.cv_kwargs:
            raise ValueError(
                "Attempting to initialize CVBackend with a static MIP is not supported. "
                "Please provide the intended resolution through the index"
            )

        self._set_cv_defaults()
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

                    CloudVolume(  # pylint: disable=no-member
                        self.path, info=new_info
                    ).commit_info()
                    _info_cache[_info_hash_key(self.path)] = new_info
            self.info_spec.reference_path = self.path
        else:
            _get_info(self.path)

    def _set_cv_defaults(self):
        self.cv_kwargs.setdefault("bounded", False)
        self.cv_kwargs.setdefault("progress", False)
        self.cv_kwargs.setdefault("autocrop", False)
        self.cv_kwargs.setdefault("non_aligned_writes", False)
        self.cv_kwargs.setdefault("cache", False)
        self.cv_kwargs.setdefault("compress_cache", False)
        self.cv_kwargs.setdefault("compress", True)
        self.cv_kwargs.setdefault("cdn_cache", False)
        self.cv_kwargs.setdefault("fill_missing", True)
        self.cv_kwargs.setdefault("delete_black_uploads", True)
        self.cv_kwargs.setdefault("agglomerate", True)

    def _get_cv_at_resolution(
        self, resolution: Vec3D
    ) -> cv.frontends.precomputed.CloudVolumePrecomputed:
        result = get_cv_cached(cloudpath=self.path, mip=tuple(resolution), **self.cv_kwargs)
        return result

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
        result = get_cv_cached(cloudpath=self.path, **self.cv_kwargs)

        dtype = result.data_type
        try:
            return getattr(torch, dtype)
        except Exception as e:
            raise ValueError(  # pylint: disable=raise-missing-from
                f"CVBackend has data_type '{dtype}',"
                " which cannot be parsed as a valid torch dtype."
            ) from e

    @property
    def num_channels(self) -> int:  # pragma: no cover
        result = get_cv_cached(cloudpath=self.path, **self.cv_kwargs)
        return result.num_channels

    @property
    def is_local(self) -> bool:  # pragma: no cover
        return self.path.startswith("file://")

    @property
    def enforce_chunk_aligned_writes(self) -> bool:  # pragma: no cover
        return not self.cv_kwargs["non_aligned_writes"]

    @enforce_chunk_aligned_writes.setter
    def enforce_chunk_aligned_writes(self, value: bool) -> None:  # pragma: no cover
        raise NotImplementedError(
            "cannot set `enforce_chunk_aligned_writes` for CVBackend directly;"
            " use `backend.with_changes(non_aligned_writes=value:bool)` instead."
        )

    @property
    def allow_cache(self) -> bool:  # pragma: no cover
        return self.cv_kwargs["cache"]

    @allow_cache.setter
    def allow_cache(self, value: Union[bool, str]) -> None:  # pragma: no cover
        raise NotImplementedError(
            "cannot set `allow_cache` for CVBackend directly;"
            " use `backend.with_changes(allow_cache=value:Union[bool, str])` instead."
        )

    @property
    def use_compression(self) -> bool:  # pragma: no cover
        return self.cv_kwargs["cache"]

    @use_compression.setter
    def use_compression(self, value: bool) -> None:  # pragma: no cover
        raise NotImplementedError(
            "cannot set `use_compression` for CVBackend directly;"
            " use `backend.with_changes(use_compression=value:bool)` instead."
        )

    def clear_cache(self) -> None:  # pragma: no cover
        info = _get_info(self.path)
        for scale in info["scales"]:
            res = Vec3D[float](*scale["resolution"])
            self._get_cv_at_resolution(res).cache.flush()

    def read(self, idx: VolumetricIndex) -> torch.Tensor:
        # Data out: bcxyz
        cvol = self._get_cv_at_resolution(idx.resolution)
        data_raw = cvol[idx.to_slices()]

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

        cvol = self._get_cv_at_resolution(idx.resolution)
        slices = idx.to_slices()
        # Enable autocrop for writes only
        cvol.autocrop = True

        cvol[slices] = data_final
        cvol.autocrop = False

    def with_changes(self, **kwargs) -> CVBackend:
        """Currently untyped. Supports:
        "name" = value: str
        "allow_cache" = value: Union[bool, str]
        "use_compression" = value: str
        "enforce_chunk_aligned_writes" = value: bool
        "voxel_offset_res" = (voxel_offset, resolution): Tuple[IntVec3D, Vec3D]
        "chunk_size_res" = (chunk_size, resolution): Tuple[IntVec3D, Vec3D]
        """
        assert self.info_spec is not None

        info_spec = deepcopy(self.info_spec)
        cv_kwargs = deepcopy(self.cv_kwargs)

        implemented_keys = [
            "name",
            "allow_cache",
            "use_compression",
            "enforce_chunk_aligned_writes",
            "voxel_offset_res",
            "chunk_size_res",
        ]
        keys_to_kwargs = {"name": "path"}
        keys_to_infospec_fn = {
            "voxel_offset_res": info_spec.set_voxel_offset,
            "chunk_size_res": info_spec.set_chunk_size,
        }
        keys_to_cv_kwargs = {
            "allow_cache": "cache",
            "use_compression": "compress",
            "enforce_chunk_aligned_writes": "non_aligned_writes",
        }
        keys_to_reverse = ["enforce_chunk_aligned_writes"]
        evolve_kwargs = {}
        for k, v in kwargs.items():
            if k not in implemented_keys:
                raise KeyError(f"key {k} received, expected one of {implemented_keys}")
            if k in keys_to_cv_kwargs:
                if k in keys_to_reverse:
                    v = not v
                cv_kwargs[keys_to_cv_kwargs[k]] = v
            if k in keys_to_kwargs:
                evolve_kwargs[keys_to_kwargs[k]] = v
            if k in keys_to_infospec_fn:
                keys_to_infospec_fn[k](v)

        return attrs.evolve(
            self,
            **evolve_kwargs,
            info_spec=info_spec,
            cv_kwargs=cv_kwargs,
            on_info_exists="overwrite",
        )

    def get_voxel_offset(self, resolution: Vec3D) -> IntVec3D:
        cvol = get_cv_cached(cloudpath=self.path, mip=tuple(resolution), **self.cv_kwargs)
        return IntVec3D(*cvol.voxel_offset)

    def get_chunk_size(self, resolution: Vec3D) -> IntVec3D:
        cvol = get_cv_cached(cloudpath=self.path, mip=tuple(resolution), **self.cv_kwargs)
        return IntVec3D(*cvol.chunk_size)

    def get_chunk_aligned_index(  # pragma: no cover
        self, idx: VolumetricIndex, mode: Literal["expand", "shrink", "round"]
    ) -> VolumetricIndex:
        cvol = get_cv_cached(cloudpath=self.path, mip=tuple(idx.resolution), **self.cv_kwargs)
        bbox = Bbox(*tuple(zip(*((s.start, s.stop) for s in idx.to_slices()))))
        if mode == "expand":
            bbox_aligned = bbox.expand_to_chunk_size(cvol.chunk_size, cvol.voxel_offset)
        elif mode == "shrink":
            bbox_aligned = bbox.shrink_to_chunk_size(cvol.chunk_size, cvol.voxel_offset)
        elif mode == "round":
            bbox_aligned = bbox.round_to_chunk_size(cvol.chunk_size, cvol.voxel_offset)
        else:
            raise NotImplementedError(
                f"mode must be set to 'expand', 'shrink', or 'round'; received '{mode}'"
            )
        return VolumetricIndex(
            resolution=idx.resolution,
            bbox=BBox3D.from_coords(
                IntVec3D(*bbox_aligned.minpt), IntVec3D(*bbox_aligned.maxpt), idx.resolution
            ),
        )

    def assert_idx_is_chunk_aligned(self, idx: VolumetricIndex) -> None:
        """check that the idx given is chunk_aligned, and give suggestions"""
        idx_expanded = self.get_chunk_aligned_index(idx, mode="expand")
        idx_rounded = self.get_chunk_aligned_index(idx, mode="round")
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
                + f" - rounded  : {idx_rounded.pformat()}\n"
                + f" - shrunk   : {idx_shrunk.pformat()}"
            )

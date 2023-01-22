# pylint: disable=missing-docstring
from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, Literal, Optional

import attrs
import cachetools
import cloudvolume as cv
import fsspec
import numpy as np
import torch
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox
from typeguard import typechecked

from zetta_utils import builder, tensor_ops
from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D

from .. import VolumetricBackend, VolumetricIndex


def _jsonize_key(*args, **kwargs):  # pragma: no cover
    result = ""
    for e in args[1:]:
        result += json.dumps(e)
        result += "_"

    result += json.dumps(kwargs, sort_keys=True)
    return result


_cv_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=500)

# To avoid reloading info file
@cachetools.cached(_cv_cache, key=_jsonize_key)
def get_cv_cached(*args, **kwargs):
    return CloudVolume(*args, **kwargs)


@builder.register("CVBackend")
@typechecked
@attrs.mutable
class PrecomputedInfoSpec:
    reference_path: Optional[str] = None
    field_overrides: Optional[Dict[str, Any]] = None
    default_chunk_size: Optional[IntVec3D] = None
    default_voxel_offset: Optional[IntVec3D] = None
    chunk_size_map: Optional[Dict[str, IntVec3D]] = None
    voxel_offset_map: Optional[Dict[str, IntVec3D]] = None
    # ensure_scales: Optional[Iterable[int]] = None

    def make_info(self) -> Optional[Dict[str, Any]]:  # pylint: disable=too-many-branches
        if self.reference_path is None and self.field_overrides is None:
            result = None
        else:
            field_overrides = self.field_overrides
            if field_overrides is None:
                field_overrides = {}
            reference_info = {}  # type: Dict[str, Any]
            if self.reference_path is not None:
                reference_info = _get_info(self.reference_path)
            result = {**reference_info, **field_overrides}
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

            # if self.ensure_scales is not None:  # pragma: no cover
            #    raise NotImplementedError()

        return result


def _get_info(path: str) -> Dict[str, Any]:
    if not path.endswith("/info"):
        path = os.path.join(path, "info")

    with fsspec.open(path) as f:
        result = json.load(f)

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

        CloudVolume(self.path)

    def _set_cv_defaults(self):
        self.cv_kwargs.setdefault("bounded", False)
        self.cv_kwargs.setdefault("progress", False)
        self.cv_kwargs.setdefault("autocrop", False)
        self.cv_kwargs.setdefault("non_aligned_writes", False)
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
            "cannot set name for CVBackend directly; use `backend.clone()` instead"
        )

    @property
    def enforce_chunk_aligned_writes(self) -> bool:  # pragma: no cover
        return not self.cv_kwargs["non_aligned_writes"]

    @enforce_chunk_aligned_writes.setter
    def enforce_chunk_aligned_writes(self, value: bool) -> None:  # pragma: no cover
        self.cv_kwargs["non_aligned_writes"] = not value

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

    def clone(self, **kwargs) -> CVBackend:
        implemented_keys = ["name"]
        for k in kwargs:
            if k not in implemented_keys:
                raise KeyError(f"key {k} received, expected one of {implemented_keys}")
        return attrs.evolve(deepcopy(self), path=kwargs["name"])

    def get_voxel_offset(self, resolution: Vec3D) -> IntVec3D:
        cvol = get_cv_cached(cloudpath=self.path, mip=tuple(resolution), **self.cv_kwargs)
        return IntVec3D(*cvol.voxel_offset)

    def set_voxel_offset(self, voxel_offset: IntVec3D, resolution: Vec3D) -> None:
        assert self.info_spec is not None
        key = "_".join([_str(v) for v in resolution])
        if self.info_spec.voxel_offset_map is None:
            self.info_spec.voxel_offset_map = {}
        self.info_spec.voxel_offset_map[key] = voxel_offset
        self.on_info_exists = "overwrite"
        self.__attrs_post_init__()

    def get_chunk_size(self, resolution: Vec3D) -> IntVec3D:
        cvol = get_cv_cached(cloudpath=self.path, mip=tuple(resolution), **self.cv_kwargs)
        return IntVec3D(*cvol.chunk_size)

    def set_chunk_size(self, chunk_size: IntVec3D, resolution: Vec3D) -> None:
        assert self.info_spec is not None
        key = "_".join([_str(v) for v in resolution])
        if self.info_spec.chunk_size_map is None:
            self.info_spec.chunk_size_map = {}
        self.info_spec.chunk_size_map[key] = chunk_size
        self.on_info_exists = "overwrite"
        self.__attrs_post_init__()

    def get_chunk_aligned_index(  # pragma: no cover
        self, index: VolumetricIndex, mode: Literal["expand", "shrink", "round"]
    ) -> VolumetricIndex:
        cvol = get_cv_cached(cloudpath=self.path, mip=tuple(index.resolution), **self.cv_kwargs)
        bbox = Bbox(*tuple(zip(*((s.start, s.stop) for s in index.to_slices()))))
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
            resolution=index.resolution,
            bbox=BBox3D.from_coords(
                IntVec3D(*bbox_aligned.minpt), IntVec3D(*bbox_aligned.maxpt), index.resolution
            ),
        )

    def assert_idx_is_chunk_aligned(self, idx: VolumetricIndex) -> None:

        # check that the idx given is chunk_aligned, and give suggestions
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

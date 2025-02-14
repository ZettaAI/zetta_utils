# pylint: disable=missing-docstring
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Union

import attrs
import cachetools
import cloudvolume as cv
import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.exceptions import ScaleUnavailableError
from numpy import typing as npt

from zetta_utils.common import abspath, is_local
from zetta_utils.geometry import Vec3D

from ....deprecated.precomputed import InfoExistsModes, PrecomputedInfoSpec, get_info
from ... import VolumetricBackend, VolumetricIndex

_cv_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=16)
_cv_cached: Dict[str, set] = {}

IN_MEM_CACHE_NUM_BYTES_PER_CV = 128 * 1024 ** 2


# To avoid reloading info file - note that an empty provenance is passed
# since otherwise the CloudVolume's __new__ will download the provenance
# TODO: Use `assume_metadata` off of the cached info, using `get_info`.
# Cannot use regular hashkey as the resolutions used need to be tracked
def _get_cv_cached(
    path: str,
    resolution: Optional[Vec3D] = None,
    cache_bytes_limit: Optional[int] = None,
    **kwargs,
) -> cv.frontends.precomputed.CloudVolumePrecomputed:
    path_ = abspath(path)
    if cache_bytes_limit is None:
        cache_bytes_limit = IN_MEM_CACHE_NUM_BYTES_PER_CV
    if (path_, resolution) in _cv_cache:
        return _cv_cache[(path_, resolution)]
    if resolution is not None:
        try:
            result = CloudVolume(
                path_,
                info=get_info(path_),
                provenance={},
                mip=tuple(resolution),
                lru_bytes=cache_bytes_limit,
                **kwargs,
            )
        except ScaleUnavailableError as e:
            raise ScaleUnavailableError(f"{path_} - {e}") from e
    else:
        result = CloudVolume(
            path_,
            info=get_info(path_),
            provenance={},
            lru_bytes=cache_bytes_limit,
            **kwargs,
        )
    _cv_cache[(path_, resolution)] = result
    if path_ not in _cv_cached:
        _cv_cached[path_] = set()
    _cv_cached[path_].add(resolution)
    return result


def _clear_cv_cache(path: str | None = None) -> None:  # pragma: no cover
    if path is None:
        _cv_cached.clear()
        _cv_cache.clear()
        return
    path_ = abspath(path)
    resolutions = _cv_cached.pop(path_, None)
    if resolutions is not None:
        for resolution in resolutions:
            _cv_cache.pop((path_, resolution), None)


@attrs.mutable
class CVBackend(VolumetricBackend):  # pylint: disable=too-few-public-methods
    """
    Backend for peforming IO on Neuroglancer datasts using CloudVolume library.
    Read data will be a ``npt.NDArray`` in ``CXYZ`` dimension order.
    Write data is expected to be a ``npt.NDArray`` or ``np.ndarray`` in ``CXYZ``
    dimension order.
    :param path: CloudVolume path. Can be given as relative or absolute.
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
    cache_bytes_limit: Optional[int] = None

    def __attrs_post_init__(self):
        if "mip" in self.cv_kwargs:
            raise ValueError(  # pragma: no cover
                "Attempting to initialize CVBackend with a static MIP is not supported. "
                "Please provide the intended resolution through the index"
            )

        self._set_cv_defaults()
        if self.info_spec is None:
            self.info_spec = PrecomputedInfoSpec()
        overwritten = self.info_spec.update_info(self.path, self.on_info_exists)
        if overwritten:
            _clear_cv_cache(self.path)

    def _set_cv_defaults(self):
        self.cv_kwargs.setdefault("bounded", False)
        self.cv_kwargs.setdefault("progress", False)
        self.cv_kwargs.setdefault("autocrop", False)
        self.cv_kwargs.setdefault("non_aligned_writes", False)
        self.cv_kwargs.setdefault("cache", False)
        self.cv_kwargs.setdefault("compress_cache", False)
        self.cv_kwargs.setdefault("compress", None)
        self.cv_kwargs.setdefault("cdn_cache", False)
        self.cv_kwargs.setdefault("fill_missing", True)
        self.cv_kwargs.setdefault("delete_black_uploads", True)
        self.cv_kwargs.setdefault("agglomerate", True)
        self.cv_kwargs.setdefault("lru_encoding", "raw")

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
    def dtype(self) -> np.dtype:
        result = _get_cv_cached(
            self.path, cache_bytes_limit=self.cache_bytes_limit, **self.cv_kwargs
        )

        return np.dtype(result.data_type)

    @property
    def num_channels(self) -> int:  # pragma: no cover
        result = _get_cv_cached(
            self.path, cache_bytes_limit=self.cache_bytes_limit, **self.cv_kwargs
        )
        return result.num_channels

    @property
    def is_local(self) -> bool:  # pragma: no cover
        return is_local(self.path)

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
        return self.cv_kwargs["compress"]

    @use_compression.setter
    def use_compression(self, value: bool) -> None:  # pragma: no cover
        raise NotImplementedError(
            "cannot set `use_compression` for CVBackend directly;"
            " use `backend.with_changes(use_compression=value:bool)` instead."
        )

    def clear_disk_cache(self) -> None:  # pragma: no cover
        info = get_info(self.path)
        for scale in info["scales"]:
            res = Vec3D[float](*scale["resolution"])
            _get_cv_cached(
                self.path,
                resolution=res,
                cache_bytes_limit=self.cache_bytes_limit,
                **self.cv_kwargs,
            ).cache.flush()

    def clear_cache(self) -> None:  # pragma: no cover
        _clear_cv_cache(self.path)

    def read(self, idx: VolumetricIndex) -> npt.NDArray:
        # Data out: cxyz
        cvol = _get_cv_cached(
            self.path, idx.resolution, cache_bytes_limit=self.cache_bytes_limit, **self.cv_kwargs
        )
        data_raw = cvol[idx.to_slices()]

        result = np.transpose(data_raw, (3, 0, 1, 2))
        return np.array(result)

    def write(self, idx: VolumetricIndex, data: npt.NDArray):
        # Data in: cxyz
        # Write format: xyzc (b == 1)
        if self.enforce_chunk_aligned_writes:
            self.assert_idx_is_chunk_aligned(idx)

        if data.size == 1 and len(data.shape) == 1:
            data_final = data[0]
        elif len(data.shape) == 4:
            data_final = np.transpose(data, (1, 2, 3, 0))
        else:
            raise ValueError(
                "Data written to CloudVolume backend must be in `cxyz` dimension format, "
                f"but got a tensor of with ndim == {data.ndim}"
            )

        cvol = _get_cv_cached(
            self.path, idx.resolution, cache_bytes_limit=self.cache_bytes_limit, **self.cv_kwargs
        )
        slices = idx.to_slices()
        # Enable autocrop for writes only
        cvol.autocrop = True

        if (cvol.dtype == "uint64") and (data_final.dtype == np.int64):
            if data_final.min() < np.int64(0):
                raise ValueError("Attempting to write negative values to a uint64 CloudVolume")
            data_final = data_final.astype(np.uint64)
        cvol[slices] = data_final
        cvol.autocrop = False

    def with_changes(self, **kwargs) -> CVBackend:
        """Currently untyped. Supports:
        "name" = value: str
        "use_compression" = value: str
        "enforce_chunk_aligned_writes" = value: bool
        "voxel_offset_res" = (voxel_offset, resolution): Tuple[Vec3D[int], Vec3D]
        "chunk_size_res" = (chunk_size, resolution): Tuple[Vec3D[int], Vec3D]
        "dataset_size_res" = (dataset_size, resolution): Tuple[Vec3D[int], Vec3D]
        "allow_cache" = value: Union[bool, str]
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
            "dataset_size_res",
        ]
        keys_to_kwargs = {"name": "path"}
        keys_to_infospec_fn = {
            "voxel_offset_res": info_spec.set_voxel_offset,
            "chunk_size_res": info_spec.set_chunk_size,
            "dataset_size_res": info_spec.set_dataset_size,
        }
        keys_to_cv_kwargs = {
            "use_compression": "compress",
            "enforce_chunk_aligned_writes": "non_aligned_writes",
        }
        keys_to_reverse = ["enforce_chunk_aligned_writes"]
        evolve_kwargs = {}
        for k, v in kwargs.items():
            if k not in implemented_keys:
                raise KeyError(f"key `{k}` received, expected one of `{implemented_keys}`")
            if k in keys_to_cv_kwargs:
                if k in keys_to_reverse:
                    v = not v
                cv_kwargs[keys_to_cv_kwargs[k]] = v
            if k in keys_to_kwargs:
                evolve_kwargs[keys_to_kwargs[k]] = v
            if k in keys_to_infospec_fn:
                keys_to_infospec_fn[k](v)

        if "name" in kwargs:
            _clear_cv_cache(kwargs["name"])
        _clear_cv_cache(self.path)
        result = attrs.evolve(
            self,
            **evolve_kwargs,
            info_spec=info_spec,
            cv_kwargs=cv_kwargs,
            on_info_exists="overwrite",
        )
        return result

    def get_voxel_offset(self, resolution: Vec3D) -> Vec3D[int]:
        cvol = _get_cv_cached(
            self.path,
            resolution=resolution,
            cache_bytes_limit=self.cache_bytes_limit,
            **self.cv_kwargs,
        )
        return Vec3D[int](*cvol.voxel_offset)

    def get_chunk_size(self, resolution: Vec3D) -> Vec3D[int]:
        cvol = _get_cv_cached(
            self.path,
            resolution=resolution,
            cache_bytes_limit=self.cache_bytes_limit,
            **self.cv_kwargs,
        )
        return Vec3D[int](*cvol.chunk_size)

    def get_dataset_size(self, resolution: Vec3D) -> Vec3D[int]:
        cvol = _get_cv_cached(
            self.path,
            resolution=resolution,
            cache_bytes_limit=self.cache_bytes_limit,
            **self.cv_kwargs,
        )
        return Vec3D[int](*cvol.volume_size)

    def get_bounds(self, resolution: Vec3D) -> VolumetricIndex:  # pragma: no cover
        offset = self.get_voxel_offset(resolution)
        size = self.get_dataset_size(resolution)
        return VolumetricIndex.from_coords(offset, offset + size, resolution)

    def pformat(self) -> str:  # pragma: no cover
        return self.name

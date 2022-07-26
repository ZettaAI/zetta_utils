# pylint: disable=missing-docstring
import json

import numpy.typing as npt
import cachetools  # type: ignore
import cloudvolume as cv  # type: ignore
from cloudvolume import CloudVolume

from zetta_utils.data.layers.data_backends.base import BaseDataBackend
from zetta_utils.data.layers.indexers.volumetric import VolumetricIndex
from zetta_utils.typing import Vec3D


cv_cache = cachetools.LRUCache(maxsize=500)


def _jsonize_key(*args, **kwargs):  # pragma: no cover
    result = ""
    for a in args[1:]:
        result += json.dumps(a)
        result += "_"

    result += json.dumps(kwargs, sort_keys=True)
    return result


class CachedCloudVolume(CloudVolume):  # pragma: no cover # pylint: disable=too-few-public-methods
    """Caching wrapper around CloudVolume."""

    @cachetools.cached(cv_cache, key=_jsonize_key)
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)


class CVBackend(BaseDataBackend):  # pylint: disable=too-few-public-methods
    def __init__(
        self,
        **kwargs,
    ):
        if "mip" in kwargs:
            raise ValueError(
                "Attempting to initialize CVBackend with a static MIP is not supported. "
                "Please provide the intended resolution through the index"
            )

        self.kwargs = kwargs
        self.kwargs.setdefault("bounded", False)
        self.kwargs.setdefault("progress", False)
        self.kwargs.setdefault("autocrop", False)
        self.kwargs.setdefault("non_aligned_writes", False)
        self.kwargs.setdefault("cdn_cache", False)
        self.kwargs.setdefault("fill_missing", True)
        self.kwargs.setdefault("delete_black_uploads", True)
        self.kwargs.setdefault("agglomerate", True)

    def _get_cv_at_resolution(
        self, resolution: Vec3D
    ) -> cv.frontends.precomputed.CloudVolumePrecomputed:
        result = CachedCloudVolume(mip=resolution, **self.kwargs)
        return result

    def read(self, idx: VolumetricIndex, **kwargs) -> npt.NDArray:
        if len(kwargs) > 0:
            raise ValueError(f"Unsupported `kwargs`: {kwargs}")

        cvol = self._get_cv_at_resolution(idx.resolution)
        result = cvol[idx.slices]
        return result

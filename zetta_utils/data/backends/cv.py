# pylint: disable=missing-docstring
import json

import attrs
import numpy as np
import numpy.typing as npt
import cachetools  # type: ignore
import cloudvolume as cv  # type: ignore
from cloudvolume import CloudVolume

import zetta_utils as zu
from zetta_utils.data.backends.base import DataBackend
from zetta_utils.data.indexes import VolumetricIndex
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


@attrs.mutable(init=False)
class CVBackend(DataBackend[VolumetricIndex]):  # pylint: disable=too-few-public-methods
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

    def read(self, idx: VolumetricIndex) -> npt.NDArray:
        # Data out: bcxyz
        cvol = self._get_cv_at_resolution(idx.resolution)
        data_raw = cvol[idx.slices]

        result = np.expand_dims(np.transpose(data_raw, (3, 0, 1, 2)), 0)

        return result

    def write(self, idx: VolumetricIndex, value: zu.typing.Tensor):
        # Data in: bcxyz
        # Write format: xyzc (b == 1)
        value = zu.data.convert.to_np(value)
        if len(value.shape) != 5:
            raise ValueError(
                "Data written to CloudVolume backend must be in `bcxyz` dimension format, "
                f"but, got a tensor of with ndim == {value.ndim}"
            )

        if value.shape[0] != 1:
            raise ValueError(
                "Data written to CloudVolume backend must have batch size of 1, "
                f"but, got a tensor of with shape == {value.shape} (b == {value.shape[0]})"
            )
        value_final = np.transpose(value[0], (1, 2, 3, 0))

        cvol = self._get_cv_at_resolution(idx.resolution)
        cvol[idx.slices] = value_final

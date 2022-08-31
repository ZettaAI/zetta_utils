# pylint: disable=missing-docstring
import json

import attrs
import numpy as np
import torch
import cachetools  # type: ignore
import cloudvolume as cv  # type: ignore
from cloudvolume import CloudVolume
from typeguard import typechecked

from zetta_utils import tensor_ops, builder
from zetta_utils.io.backends.base import IOBackend
from zetta_utils.io.indexes import VolumetricIndex
from zetta_utils.typing import Vec3D, Tensor


cv_cache = cachetools.LRUCache(maxsize=500)


def _jsonize_key(*args, **kwargs):  # pragma: no cover
    result = ""
    for e in args[1:]:
        result += json.dumps(e)
        result += "_"

    result += json.dumps(kwargs, sort_keys=True)
    return result


class CachedCloudVolume(CloudVolume):  # pragma: no cover # pylint: disable=too-few-public-methods
    """Caching wrapper around CloudVolume."""

    @cachetools.cached(cv_cache, key=_jsonize_key)
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)


@builder.register("CVBackend")
@typechecked
@attrs.mutable(init=False)
class CVBackend(IOBackend[VolumetricIndex]):  # pylint: disable=too-few-public-methods
    """
    Backend for peforming IO on Neuroglancer datasts using CloudVolume library.
    Read data will be a ``torch.Tensor`` in ``BCXYZ`` dimension order.
    Write data is expected to be a ``torch.Tensor`` or ``np.ndarray`` in ``BCXYZ``
    dimension order.

    :param device: Device name where read tensors will reside in torch format.
    :param kwargs: Parameters that will be passed to the CloudVolume constructor.
        ``mip`` keyword must not be present in ``kwargs``, as the read resolution
        is passed in as a part of index to the backend.
    """

    def __init__(
        self,
        device: str = "cpu",
        **kwargs,
    ):
        if "mip" in kwargs:
            raise ValueError(
                "Attempting to initialize CVBackend with a static MIP is not supported. "
                "Please provide the intended resolution through the index"
            )

        self.device = device
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

    def read(self, idx: VolumetricIndex) -> torch.Tensor:
        # Data out: bcxyz
        cvol = self._get_cv_at_resolution(idx.resolution)
        data_raw = cvol[idx.slices]

        result_np = np.transpose(data_raw, (3, 0, 1, 2))
        result = tensor_ops.to_torch(result_np, device=self.device)
        return result

    def write(self, idx: VolumetricIndex, value: Tensor):
        # Data in: bcxyz
        # Write format: xyzc (b == 1)
        value = tensor_ops.convert.to_np(value)
        if len(value.shape) != 4:
            raise ValueError(
                "Data written to CloudVolume backend must be in `cxyz` dimension format, "
                f"but, got a tensor of with ndim == {value.ndim}"
            )

        value_final = np.transpose(value, (1, 2, 3, 0))

        cvol = self._get_cv_at_resolution(idx.resolution)
        cvol[idx.slices] = value_final

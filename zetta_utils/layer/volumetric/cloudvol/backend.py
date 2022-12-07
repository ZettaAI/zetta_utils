# pylint: disable=missing-docstring
import json
import os
from typing import Any, Dict, Literal, Optional

import attrs
import cachetools
import cloudvolume as cv
import fsspec
import numpy as np
import torch
from cloudvolume import CloudVolume
from typeguard import typechecked

from zetta_utils import builder, tensor_ops
from zetta_utils.tensor_typing import Tensor
from zetta_utils.typing import IntVec3D, Vec3D

from ... import LayerBackend
from .. import VolumetricIndex


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
    chunk_size: Optional[IntVec3D] = None
    # ensure_scales: Optional[Iterable[int]] = None

    def make_info(self) -> Optional[Dict[str, Any]]:
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
            if self.chunk_size is not None:
                for e in result["scales"]:
                    e["chunk_sizes"] = [self.chunk_size]

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


@builder.register("CVBackend")
@typechecked
@attrs.mutable
class CVBackend(
    LayerBackend[VolumetricIndex, torch.Tensor]
):  # pylint: disable=too-few-public-methods
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
        result = get_cv_cached(cloudpath=self.path, mip=resolution, **self.cv_kwargs)
        return result

    def read(self, idx: VolumetricIndex) -> torch.Tensor:
        # Data out: bcxyz
        cvol = self._get_cv_at_resolution(idx.resolution)
        data_raw = cvol[idx.to_slices()]

        result_np = np.transpose(data_raw, (3, 0, 1, 2))
        result = tensor_ops.to_torch(result_np)
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
        slices = idx.to_slices()
        # Enable autocrop for writes only
        cvol.autocrop = True
        cvol[slices] = value_final
        cvol.autocrop = False

    def get_name(self) -> str:  # pragma: no cover
        return self.path

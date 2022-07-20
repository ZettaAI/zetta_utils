"""Volumetric layers are referenced by [(MIP), z, x, y] and
support a `data_mip` parameter which can allow reading
raw data from a different MIP, followed by (up/down)sampling."""
from __future__ import annotations

import json
from typing import Literal, Union, Optional, Tuple
import cachetools  # type: ignore
import numpy as np
import cloudvolume as cv  # type: ignore
from cloudvolume import CloudVolume  # type: ignore
from typeguard import typechecked

import zetta_utils as zu
from zetta_utils.data.layers.common import BaseLayer

VolumetricIndex = Union[
    zu.bcube.BoundingCube,
    Tuple[zu.bcube.BoundingCube],
    Tuple[slice, slice, slice],
    Tuple[Optional[zu.bcube.VolumetricResolution], zu.bcube.BoundingCube],
    Tuple[Optional[zu.bcube.VolumetricResolution], slice, slice, slice],
]

StandardVolumetricIndex = Tuple[
    Optional[zu.bcube.VolumetricResolution], zu.bcube.BoundingCube
]


def translate_volumetric_index(  # pylint: disable=missing-docstring
    idx: VolumetricIndex,
    offset: Tuple[int, int, int],
    offset_resolution: zu.bcube.VolumetricResolution = None,
) -> VolumetricIndex:
    if isinstance(idx, zu.bcube.BoundingCube):  # [bcube] indexing
        if offset_resolution is None:
            offset_resolution = (1, 1, 1)
        result = idx.translate(offset, offset_resolution)  # type: VolumetricIndex
    elif isinstance(
        idx[-1], zu.bcube.BoundingCube
    ):  # [bcube,] or [resolution, bcube] indexing
        if offset_resolution is None:
            offset_resolution = (1, 1, 1)
        bcube = idx[-1].translate(offset, offset_resolution)
        if len(idx) == 2:
            result = (idx[0], bcube)  # type: ignore # doens't know idx[0] == resolution
        else:
            assert len(idx) == 1, "Type checker error."
            result = (bcube,)
    else:
        x_slice = slice(idx[-3].start + offset[0], idx[-3].stop + offset[0])
        y_slice = slice(idx[-2].start + offset[1], idx[-2].stop + offset[1])
        z_slice = slice(idx[-1].start + offset[2], idx[-1].stop + offset[2])

        if len(idx) == 4:  # [resolution, x_slice, y_slice, z_slice] indexing
            result = (idx[0], x_slice, y_slice, z_slice)  # type: ignore
        else:
            assert len(idx) == 3, "Type checker error."
            result = (x_slice, y_slice, z_slice)

    return result


@typechecked
def _standardize_vol_idx(
    in_idx: VolumetricIndex,
    index_resolution: Optional[zu.bcube.VolumetricResolution] = None,
) -> StandardVolumetricIndex:
    resolution = None  # type: Optional[zu.bcube.VolumetricResolution]

    if isinstance(in_idx, zu.bcube.BoundingCube):
        bcube = in_idx  # [bcube] indexing
    elif isinstance(
        in_idx[-1], zu.bcube.BoundingCube
    ):  # [bcube,] or [resolution, bcube] indexing
        bcube = in_idx[-1]
        if len(in_idx) == 2:  # [resolution, bcube]
            resolution = in_idx[0]  # type: ignore
        elif len(in_idx) != 1:  # not [bcube, ]
            raise ValueError(f"Malformed volumetric index '{in_idx}'")
    else:
        if index_resolution is None:
            raise ValueError(
                "Attempting to convert slice based index to volumetric "
                "index used without `index_resolution` provided."
            )

        if len(in_idx) == 4:  # [resolution, x_slice, y_slice, z_slice] indexing
            resolution = in_idx[0]  # type: ignore
            bcube = zu.bcube.BoundingCube(
                slices=in_idx[1:], resolution=index_resolution  # type: ignore
            )
        elif len(in_idx) == 3:  # [x_slice, y_slice, z_slice] indexing
            bcube = zu.bcube.BoundingCube(
                slices=in_idx, resolution=index_resolution  # type: ignore
            )  # pylint: disable=line-too-long

    return (resolution, bcube)


@typechecked
class VolumetricLayer(BaseLayer):
    """Volumetric Layer."""

    def __init__(
        self,
        index_resolution: Optional[zu.bcube.VolumetricResolution] = None,
        data_resolution: Optional[zu.bcube.VolumetricResolution] = None,
        interpolation_mode: Optional[zu.data.basic_ops.InterpolationMode] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_resolution = data_resolution
        self.index_resolution = index_resolution
        self.interpolation_mode = interpolation_mode

    def _write(self, idx: VolumetricIndex, value) -> None:
        # self._write_volume(idx, value)
        raise NotImplementedError()

    def __getitem__(self, in_idx: VolumetricIndex) -> zu.typing.Array:
        idx = _standardize_vol_idx(in_idx, index_resolution=self.index_resolution)
        result = self.read(idx)
        return result

    def _read(self, idx: StandardVolumetricIndex) -> zu.typing.Array:
        """Handles data resolution redirection and rescaling."""
        read_resolution, bcube = idx
        # If the user didn't specify read resolution in the idnex, we
        # take the data resolution as the read resolution. If data
        # resolution is also None, we take the indexing resolution as
        # the read resolution.
        if read_resolution is None:
            if self.data_resolution is not None:
                read_resolution = self.data_resolution
            else:
                read_resolution = self.index_resolution

        # If data resolution is not specified for the volume, data will be
        # fetched from the read resolution.
        data_resolution = self.data_resolution
        if data_resolution is None:
            data_resolution = read_resolution

        final_idx = (data_resolution, bcube)
        vol_data = self._read_volume(final_idx)

        if data_resolution != read_resolution:  # output rescaling needed
            if self.interpolation_mode is None:
                raise ValueError(
                    "`data_resolution` differs from `read_resolution`, but "
                    "`interpolation_method` is not set for the layer"
                )
            raise NotImplementedError()
        # else:
        result = vol_data

        return result

    def _read_volume(self, idx: StandardVolumetricIndex) -> zu.typing.Array:
        raise NotImplementedError(
            "`_read_volume` method not implemented for a volumetric layer type."
        )

    def _write_volume(
        self, idx: StandardVolumetricIndex, value: zu.typing.Array
    ) -> None:
        raise NotImplementedError(
            "`_write` method not implemented for a volumetric layer type."
        )


def _jsonize_key(*args, **kwargs):
    result = ""
    for a in args[1:]:
        result += json.dumps(a)
        result += "_"

    result += json.dumps(kwargs, sort_keys=True)
    return result


def _cv_is_cached(*kargs, **kwargs):
    key = _jsonize_key(*kargs, **kwargs)
    return key in cv_cache


cv_cache = cachetools.LRUCache(maxsize=500)


class CachedCloudVolume(CloudVolume):
    """Caching wrapper around CloudVolume."""

    @cachetools.cached(cv_cache, key=_jsonize_key)
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)


VolumetricDimOrder = Literal["xyzc", "cxyz"]


@typechecked
class CVLayer(VolumetricLayer):
    """CloudVolume volumetric layer implementation."""

    def __init__(
        self,
        cv_params: dict,
        dim_order: VolumetricDimOrder = "cxyz",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert "mip" not in cv_params
        self.cv_params = cv_params
        self.cv_params.setdefault("bounded", False)
        self.cv_params.setdefault("progress", False)
        self.cv_params.setdefault("autocrop", False)
        self.cv_params.setdefault("non_aligned_writes", False)
        self.cv_params.setdefault("cdn_cache", False)
        self.cv_params.setdefault("fill_missing", True)
        self.cv_params.setdefault("delete_black_uploads", True)
        self.cv_params.setdefault("agglomerate", True)
        self.dim_order = dim_order

    def _get_cv_at_resolution(
        self, resolution: zu.bcube.VolumetricResolution
    ) -> cv.frontends.precomputed.CloudVolumePrecomputed:  # CloudVolume is not a CloudVolume # pylint: disable=line-too-long
        result = CloudVolume(mip=resolution, **self.cv_params)
        return result

    def _write_volume(
        self, idx: StandardVolumetricIndex, value: zu.typing.Array
    ) -> None:
        raise NotImplementedError()

    def _read_volume(self, idx: StandardVolumetricIndex) -> zu.typing.Array:
        resolution, bcube = idx
        if resolution is None:
            raise ValueError(
                "Attempting to read from a CVLayer without "
                "specifying read resolution."
            )
        cvol = self._get_cv_at_resolution(resolution)
        x_range = bcube.get_x_range(x_res=resolution[0])
        y_range = bcube.get_y_range(y_res=resolution[1])
        z_range = bcube.get_z_range(z_res=resolution[2])
        raw_result = cvol[
            x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]
        ]
        if self.dim_order == "cxyz":
            result = np.transpose(raw_result, (3, 0, 1, 2))
        elif self.dim_order == "xyzc":
            result = raw_result
        return result

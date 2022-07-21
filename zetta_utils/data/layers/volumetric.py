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
from zetta_utils.data.layers.common import Layer
from zetta_utils.typing import Array, Slice3D, Vec3D

VolumetricIndex = Union[
    zu.bcube.BoundingCube,
    Slice3D,
    Tuple[Optional[Vec3D], zu.bcube.BoundingCube],
    Tuple[Optional[Vec3D], slice, slice, slice],
]
StandardVolumetricIndex = Tuple[Optional[Vec3D], zu.bcube.BoundingCube]


def translate_volumetric_index(  # pylint: disable=missing-docstring
    idx: VolumetricIndex,
    offset: Tuple[int, int, int],
    offset_resolution: Optional[Vec3D] = None,
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
            raise TypeError  # pragma: no cover
    else:
        if offset_resolution is not None:
            raise ValueError(
                "`ofset_resoluiton` is not supported for slice-based index translation"
            )

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
    index_resolution: Optional[Vec3D] = None,
) -> StandardVolumetricIndex:
    resolution = None  # type: Optional[Vec3D]

    if isinstance(in_idx, zu.bcube.BoundingCube):
        bcube = in_idx  # [bcube] indexing
    elif isinstance(
        in_idx[-1], zu.bcube.BoundingCube
    ):  # [bcube,] or [resolution, bcube] indexing
        bcube = in_idx[-1]
        if len(in_idx) == 2:  # [resolution, bcube]
            resolution = in_idx[0]  # type: ignore
        else:
            raise TypeError  # pragma: no cover
    else:
        if index_resolution is None:
            raise ValueError(
                "Attempting standardize slice based index to volumetric "
                "index without providing `index_resolution`."
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


DimOrder3D = Literal["cxy", "bcxy", "cxyz", "bcxyz"]


@typechecked
class VolumetricLayer(Layer):
    """Volumetric Layer."""

    def __init__(
        self,
        index_resolution: Optional[Vec3D] = None,
        data_resolution: Optional[Vec3D] = None,
        interpolation_mode: Optional[zu.data.basic_ops.InterpolationMode] = None,
        dim_order: DimOrder3D = "bcxyz",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_resolution = data_resolution
        self.index_resolution = index_resolution
        self.interpolation_mode = interpolation_mode
        self.dim_order = dim_order

    def _write(self, idx: VolumetricIndex, value) -> None:
        # self._write_volume(idx, value)
        raise NotImplementedError  # pragma: no cover

    def __getitem__(self, in_idx: VolumetricIndex) -> Array:
        idx = _standardize_vol_idx(in_idx, index_resolution=self.index_resolution)
        result = self.read(idx)
        return result

    def _read(self, idx: StandardVolumetricIndex) -> Array:
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
                if self.index_resolution is None:
                    raise RuntimeError(
                        "Attempting to read with neither read resolution specified "
                        "nor `index_resolution` specified for the volume."
                    )
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
                raise RuntimeError(
                    "`data_resolution` differs from `read_resolution`, but "
                    f"`interpolation_mode` == None for {self}."
                )
            if "b" not in self.dim_order:
                raise RuntimeError(
                    "Missing batch dimension: `data_resolution` differs from "
                    "`read_resolution`, but "
                    f"`dim_order` == '{self.dim_order}' for {self}. "
                    f"Consider using '{'b' + self.dim_order}' to support interpolation."
                )

            if self.dim_order.endswith("xyz"):
                scale_factor = tuple(
                    data_resolution[i] / read_resolution[i] for i in range(3)
                )
            else:
                assert self.dim_order.endswith("xy")
                scale_factor = tuple(
                    data_resolution[i] / read_resolution[i] for i in range(2)
                )
            result = zu.data.basic_ops.interpolate(
                vol_data,
                scale_factor=scale_factor,
                mode=self.interpolation_mode,
            )
        else:
            result = vol_data

        # import pdb; pdb.set_trace()
        # print(result.shape)
        return result

    def _read_volume(self, idx: StandardVolumetricIndex) -> Array:
        raise NotImplementedError(  # pragma: no cover
            "`_read_volume` method not implemented for a volumetric layer type."
        )

    def _write_volume(self, idx: StandardVolumetricIndex, value: Array) -> None:
        raise NotImplementedError(  # pragma: no cover
            "`_write` method not implemented for a volumetric layer type."
        )


def _jsonize_key(*args, **kwargs):
    result = ""
    for a in args[1:]:  # pragma: no cover
        result += json.dumps(a)
        result += "_"

    result += json.dumps(kwargs, sort_keys=True)
    return result


def _cv_is_cached(*kargs, **kwargs):  # pragma: no cover
    key = _jsonize_key(*kargs, **kwargs)
    return key in cv_cache


cv_cache = cachetools.LRUCache(maxsize=500)


class CachedCloudVolume(CloudVolume):
    """Caching wrapper around CloudVolume."""

    @cachetools.cached(cv_cache, key=_jsonize_key)
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)


@typechecked
class CVLayer(VolumetricLayer):
    """CloudVolume volumetric layer implementation."""

    def __init__(
        self,
        cv_params: dict,
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

    def _get_cv_at_resolution(
        self, resolution: Vec3D
    ) -> cv.frontends.precomputed.CloudVolumePrecomputed:  # CloudVolume is not a CloudVolume # pylint: disable=line-too-long
        result = CachedCloudVolume(mip=resolution, **self.cv_params)
        return result

    def _write_volume(self, idx: StandardVolumetricIndex, value: Array) -> None:
        raise NotImplementedError()  # pragma: no cover

    def _read_volume(self, idx: StandardVolumetricIndex) -> Array:
        resolution, bcube = idx
        assert resolution is not None  # pragma: no test

        cvol = self._get_cv_at_resolution(resolution)
        slices = bcube.get_slices(resolution)
        raw_result = cvol[slices]

        if self.dim_order.endswith("cxy"):
            result = np.transpose(raw_result, (3, 0, 1, 2))
            if result.shape[-1] != 1:
                raise RuntimeError(
                    "Attempting to read multiple sections while "
                    "the layer is configured to use 'cxy' (no z) "
                    "dimension order."
                )
            result = result[..., 0]
        elif self.dim_order.endswith("cxyz"):
            result = np.transpose(raw_result, (3, 0, 1, 2))

        if self.dim_order.startswith("b"):
            result = np.expand_dims(result, 0)

        return result

    def __repr__(self):
        return f"CVLayer(cloudpath='{self.cv_params['cloudpath']}')"

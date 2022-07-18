"""Volumetric layers are referenced by [(MIP), z, x, y] and
support a `data_mip` parameter which can allow reading
raw data from a different MIP, followed by (up/down)sampling."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Union, Optional
import numpy as np
import cachetools  # type: ignore
from cloudvolume import CloudVolume  # type: ignore

import zetta_utils as zu
from .common import BaseLayer


@dataclass
class VolumetricIndex:
    """Index for volumetric layers."""

    xy_res: Optional[int]
    bcube: zu.bcube.BoundingCube


def _convert_to_vol_idx(
    in_idx: Union[VolumetricIndex, list], index_xy_res: Optional[int] = None
):
    if isinstance(in_idx, VolumetricIndex):
        return in_idx

    bcube = None
    xy_res = None

    if isinstance(
        in_idx[-1], zu.bcube.BoundingCube
    ):  # [bcube] or [xy_res, bcube] indexing
        bcube = in_idx[-1]
        if len(in_idx) == 2:  # [xy_res, bcube]
            xy_res = in_idx[0]
        elif len(in_idx) != 1:  # not [bcube]
            raise ValueError(f"Malformed volumetric index '{in_idx}'")
    elif isinstance(
        in_idx[-1], str
    ):  # [xy_res, start_coord, end_coord] or [start_coord, end_coord] indexing
        raise NotImplementedError()
    else:
        if index_xy_res is None:
            raise ValueError("Slice indexing used without `index_xy_res` provided.")

        if len(in_idx) == 4:  # [xy_res, z_slice, y_slice, x_slice] indexing
            xy_res = in_idx[0]
            bcube = zu.bcube.BoundingCube(slices=in_idx[1:], xy_res=index_xy_res)
        elif len(in_idx) == 3:  # [z_slice, y_slice, x_slice] or
            bcube = zu.bcube.BoundingCube(slices=in_idx, xy_res=index_xy_res)

    # mypy doesn't see that bcube is always not None
    return VolumetricIndex(bcube=bcube, xy_res=xy_res)  # type: ignore


class VolumetricLayer(BaseLayer):
    """Volumetric Layer."""

    def __init__(
        self,
        index_xy_res: int,
        data_xy_res: Optional[int] = None,
        rescaling_method: Optional[
            Union[
                Literal["img"],
                Literal["mask"],
                Literal["field"],
            ]
        ] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_xy_res = data_xy_res
        self.index_xy_res = index_xy_res
        self.rescaling_method = rescaling_method

    def _write(self, idx: Union[VolumetricIndex, list], value):
        # self._write_volume(idx, value)
        raise NotImplementedError()

    def __getitem__(self, in_idx: Union[VolumetricIndex, list]):
        idx = _convert_to_vol_idx(in_idx, index_xy_res=self.index_xy_res)
        result = self.read(idx)
        return result

    def _read(self, idx: VolumetricIndex):
        """Handles data xy_res redirection and rescaling."""
        read_xy_res = idx.xy_res
        # If the user didn't specify read resolution in the idnex, we
        # take the data resolution as the read resolution. If data
        # resolution is also None, we take the indexing resolution as
        # the read resolution.
        if read_xy_res is None:
            if self.data_xy_res is not None:
                read_xy_res = self.data_xy_res
            else:
                read_xy_res = self.index_xy_res

        # If data resolution is not specified for the volume, data will be
        # fetched from the read resolution.
        data_xy_res = self.data_xy_res
        if data_xy_res is None:
            data_xy_res = read_xy_res

        idx.xy_res = data_xy_res
        vol_data = self._read_volume(idx)

        if data_xy_res != read_xy_res:  # output rescaling needed
            raise NotImplementedError()
        # else:
        result = vol_data

        return result

    def _read_volume(self, idx: VolumetricIndex):
        raise NotImplementedError(
            "`_read_volume` method not implemented for a volumetric layer type."
        )

    def _write_volume(self, idx: VolumetricIndex, value):
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


class CVLayer(VolumetricLayer):
    """CloudVolume volumetric layer implementation."""

    def __init__(
        self,
        cv_params: dict,
        z_res: int,
        return_format: Union[Literal["zcyx"], Literal["xyzc"]] = "zcyx",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.z_res = z_res
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
        self.return_format = return_format

    def _get_cv_at_xy_res(self, xy_res):
        result = CloudVolume(mip=[xy_res, xy_res, self.z_res], **self.cv_params)
        return result

    def _write_volume(self, idx: VolumetricIndex, value):
        raise NotImplementedError()

    def _read_volume(self, idx: VolumetricIndex):
        cv = self._get_cv_at_xy_res(idx.xy_res)
        x_range = idx.bcube.get_x_range(xy_res=idx.xy_res)
        y_range = idx.bcube.get_y_range(xy_res=idx.xy_res)
        z_range = idx.bcube.get_z_range()
        result = cv[
            x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]
        ]

        if self.return_format == "zcyx":
            result = np.transpose(result, (2, 3, 1, 0))
        elif self.return_format == "xyzc":
            pass
        else:
            raise ValueError(f"Unsupported `return_format`: '{self.return_format}'")

        return result

# pylint: disable=missing-docstring
from __future__ import annotations

import copy
import math
import os
from typing import Any, Literal, Sequence, Union

import attrs
import cachetools
from cachetools.keys import hashkey
from cloudfiles import CloudFile
from cloudvolume import CloudVolume
from typeguard import typechecked

from zetta_utils.common import abspath, is_local
from zetta_utils.geometry.bbox import BBox3D
from zetta_utils.geometry.vec import Vec3D

_info_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=500)
_info_hash_key = hashkey


# wrapper to cache using absolute paths with '/info'.
# invalidates the cached infofile if the infofile is local and has since been deleted.
def get_info(path: str) -> dict[str, Any]:
    info_path = _to_info_path(path)
    if is_local(info_path):
        if not CloudFile(info_path).exists():
            _info_cache.pop(_info_hash_key(info_path), None)
    return _get_info_from_info_path(info_path)


PrecomputedVolumeDType = Literal[
    "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "float32"
]
PrecomputedAnnotationDType = Literal["POINT", "LINE", "AXIS_ALIGNED_BOUNDING_BOX", "ELLIPSOID"]
PrecomputedDType = Union[PrecomputedVolumeDType, PrecomputedAnnotationDType]

NON_INHERITABLE_SCALE_KEYS = [
    "sharding",
    "encoding",
    "voxel_offset",
    "size",
    "chunk_sizes",
    "resolution",
    "key",
]


@attrs.mutable
class InfoSpecParams:
    type: Literal["image", "segmentation", "annotation"]
    encoding: str
    scales: Sequence[Sequence[float]]
    chunk_size: Sequence[int]
    data_type: PrecomputedVolumeDType
    num_channels: int
    bbox: BBox3D
    extra_scale_data: dict | None

    def __attrs_post_init__(self):
        if len(self.scales) == 0:
            raise ValueError("At least one scale must be provided")

    @typechecked
    @classmethod
    def from_optional_reference(
        cls,
        scales: Sequence[Sequence[float]],
        reference_path: str | None = None,
        inherit_all_params: bool = False,
        type: Literal["image", "segmentation"] | None = None,  # pylint: disable=redefined-builtin
        data_type: PrecomputedVolumeDType | None = None,
        chunk_size: Sequence[int] | None = None,
        num_channels: int | None = None,
        encoding: str | None = None,
        bbox: BBox3D | None = None,
        extra_scale_data: dict | None = None,
    ):  # pylint: disable=too-many-branches,too-many-boolean-expressions # trivial branches for exception text
        """
        Create an InfoSpecParams instance from a reference path.

        :param reference_path: Path to the reference info file. Note that `scales`
            cannot be inherited.
        :param inherit_all_params: If True, inherit all unspecified parameters from the
            reference info file. If False, only bbox will be inherited.
        :param scales: Sequence of scales to be ensured in the info. Must be
            non-empty.
        :param type: Precomputed volume type (e.g., "image", "segmentation"). If
            inherit_all_params is True, this can be None.
        :param data_type: Data type of the volume (e.g., "uint8", "int16"). If
            inherit_all_params is True, this can be None.
        :param chunk_size: Size of the chunks. If inherit_all_params is True, this can
            be None.
        :param num_channels: Number of channels. If inherit_all_params is True, this
            can be None.
        :param encoding: Encoding type. If inherit_all_params is True, this can be None.
        :param bbox: Bounding box corresponding to the bounds of the dataset.
            If `None`, will be inherited from the reference.
        :param extra_scale_data: Extra information to put into every scale. Not inherited
            from reference.
        """

        if reference_path is None:
            if (
                type is None
                or data_type is None
                or chunk_size is None
                or num_channels is None
                or encoding is None
                or bbox is None
            ):
                missing_params = []
                if type is None:
                    missing_params.append("type")
                if data_type is None:
                    missing_params.append("data_type")
                if chunk_size is None:
                    missing_params.append("chunk_size")
                if num_channels is None:
                    missing_params.append("num_channels")
                if encoding is None:
                    missing_params.append("encoding")
                if bbox is None:
                    missing_params.append("bbox")

                raise ValueError(
                    f"When 'reference_path' is None, the following parameters must be "
                    f"specified and cannot be None: {', '.join(missing_params)}"
                )
            if inherit_all_params:
                raise ValueError(
                    "When `reference_path` is None, `inherit_all_params` must be False"
                )

            return cls(
                type=type,
                encoding=encoding,
                scales=scales,
                chunk_size=chunk_size,
                data_type=data_type,
                num_channels=num_channels,
                bbox=bbox,
                extra_scale_data=extra_scale_data,
            )
        else:
            return cls.from_reference(
                reference_path=reference_path,
                scales=scales,
                type=type,
                data_type=data_type,
                chunk_size=chunk_size,
                num_channels=num_channels,
                encoding=encoding,
                bbox=bbox,
                extra_scale_data=extra_scale_data,
                inherit_all_params=inherit_all_params,
            )

    @typechecked
    @staticmethod
    def from_reference(
        reference_path: str,
        scales: Sequence[Sequence[float]],
        type: Literal["image", "segmentation"] | None = None,  # pylint: disable=redefined-builtin
        data_type: PrecomputedVolumeDType | None = None,
        chunk_size: Sequence[int] | None = None,
        num_channels: int | None = None,
        encoding: str | None = None,
        bbox: BBox3D | None = None,
        extra_scale_data: dict | None = None,
        inherit_all_params: bool = False,
    ):  # pylint: disable=too-many-branches # trivial branches for exception text
        """
        Create an InfoSpecParams instance from a reference path.

        :param reference_path: Path to the reference info file. Note that `scales`
            cannot be inherited.
        :param scales: Sequence of scales to be added to be used in the info. Must be
            non-empty.
        :param type: Precomputed volume type (e.g., "image", "segmentation"). If
            inherit_all_params is True, this can be None.
        :param data_type: Data type of the volume (e.g., "uint8", "int16"). If
            inherit_all_params is True, this can be None.
        :param chunk_size: Size of the chunks. If inherit_all_params is True, this can
            be None.
        :param num_channels: Number of channels. If inherit_all_params is True, this
            can be None.
        :param encoding: Encoding type. If inherit_all_params is True, this can be None.
        :param bbox: Bounding box corresponding to the bounds of the dataset.
            If `None`, will be inherited from the reference.
        :param inherit_all_params: If True, inherit all unspecified parameters from the
            reference info file. If False, only bbox will be inherited.
        :param extra_scale_data: Extra information to put into every scale. Not inherited
            from reference.

        :raises ValueError: If scales is empty or if not all of voxel_offset, size, and
            bounds_resolution are specified when inherit_all_params is False.
        :raises ValueError: If required parameters are missing when inherit_all_params
            is False.
        """
        reference_info = get_info(reference_path)
        reference_scale = reference_info["scales"][0]

        if bbox is None:
            bbox = BBox3D.from_coords(
                start_coord=reference_scale["voxel_offset"],
                end_coord=Vec3D(*reference_scale["size"])
                + Vec3D(*reference_scale["voxel_offset"]),
                resolution=reference_scale["resolution"],
            )

        if inherit_all_params:
            if type is None:
                type = reference_info["type"]
            if data_type is None:
                data_type = reference_info["data_type"]
            if num_channels is None:
                num_channels = reference_info["num_channels"]

            if encoding is None:
                encoding = reference_scale["encoding"]
            if chunk_size is None:
                chunk_size = reference_scale["chunk_sizes"][0]

            if extra_scale_data is None:
                extra_scale_data = {}
            extra_scale_data = {
                **{
                    k: v for k, v in reference_scale.items() if k not in NON_INHERITABLE_SCALE_KEYS
                },
                **extra_scale_data,
            }
        else:
            missing_params = []
            if type is None:
                missing_params.append("type")
            if data_type is None:
                missing_params.append("data_type")
            if num_channels is None:
                missing_params.append("num_channels")
            if encoding is None:
                missing_params.append("encoding")
            if chunk_size is None:
                missing_params.append("chunk_size")

            if missing_params:
                raise ValueError(
                    f"The following parameters must be provided when 'inherit_all_params' "
                    f"is False: {', '.join(missing_params)}"
                )
        assert type is not None
        assert data_type is not None
        assert num_channels is not None
        assert encoding is not None
        assert chunk_size is not None
        assert bbox is not None

        return InfoSpecParams(
            type=type,
            encoding=encoding,
            scales=scales,
            chunk_size=chunk_size,
            data_type=data_type,
            num_channels=num_channels,
            extra_scale_data=extra_scale_data,
            bbox=bbox,
        )


@typechecked
@attrs.mutable
class PrecomputedInfoSpec:
    info_path: str | None = None
    info_spec_params: InfoSpecParams | None = None

    def __attrs_post_init__(self):
        if (self.info_path is None and self.info_spec_params is None) or (
            self.info_path is not None and self.info_spec_params is not None
        ):
            raise ValueError("Exactly one of `info_path`/`info_spec_params` must be provided")

    def make_info(self) -> dict:
        if self.info_path is not None:
            return get_info(self.info_path)
        else:
            assert self.info_spec_params is not None
            result: dict = {
                "num_channels": self.info_spec_params.num_channels,
                "data_type": self.info_spec_params.data_type,
                "type": self.info_spec_params.type,
                "scales": [],
            }
            for scale_resolution in self.info_spec_params.scales:
                scale_dict: dict = {
                    "chunk_sizes": [list(self.info_spec_params.chunk_size)],
                    "encoding": self.info_spec_params.encoding,
                    "resolution": list(scale_resolution),
                    "key": res_to_key(scale_resolution),
                }
                if self.info_spec_params.extra_scale_data is not None:
                    scale_dict = {**scale_dict, **self.info_spec_params.extra_scale_data}

                scale_dict["size"] = [
                    math.ceil(k / m)
                    for k, m in zip(self.info_spec_params.bbox.shape, scale_resolution)
                ]
                scale_dict["voxel_offset"] = [
                    k / m for k, m in zip(self.info_spec_params.bbox.start, scale_resolution)
                ]

                for i in range(len(scale_dict["voxel_offset"])):
                    if not is_integer_within_eps(scale_dict["voxel_offset"][i]):
                        scale_dict["voxel_offset"][i] = math.floor(scale_dict["voxel_offset"][i])
                    else:
                        scale_dict["voxel_offset"][i] = int(scale_dict["voxel_offset"][i])

                result["scales"].append(scale_dict)

            return result

    def update_info(self, path: str, overwrite: bool, keep_existing_scales: bool) -> bool:
        """
        Update infofile at `path`. Returns True if there was a write, else returns False.
        """
        try:
            existing_info = get_info(path)
        except FileNotFoundError:
            existing_info = None

        new_info = self.make_info()

        if new_info is not None:
            if existing_info is not None:
                if keep_existing_scales:
                    if (new_info["data_type"] != existing_info["data_type"]) or (
                        new_info["num_channels"] != existing_info["num_channels"]
                    ):
                        raise RuntimeError(
                            "Attempting to keep existing scales while 'data_type' or "
                            "'num_channels' have changed in the info file. "
                            "Consider setting `keep_existing_scales` to False."
                        )
                    new_info["scales"] = _merge_and_sort_scales(
                        existing_info["scales"], new_info["scales"]
                    )

                if not overwrite:
                    existing_scales_changed = any(
                        not (e in new_info["scales"]) for e in existing_info["scales"]
                    )
                    if existing_scales_changed:
                        raise RuntimeError(
                            f"New info is not a pure extension of the info existing at '{path}' "
                            "while `on_info_exists` is set to 'expect_same'. Some scales present "
                            f"in `{path}` would be overwritten."
                        )
                    existing_info_no_scales = copy.deepcopy(existing_info)
                    del existing_info_no_scales["scales"]
                    new_info_no_scales = copy.deepcopy(new_info)
                    del new_info_no_scales["scales"]
                    non_scales_changed = existing_info_no_scales != new_info_no_scales
                    if non_scales_changed:
                        raise RuntimeError(
                            f"New info is not a pure extension of the info existing at '{path}' "
                            "while `on_info_exists` is set to 'expect_same'. Some non-scale keys "
                            f"in `{path}` would be overwritten."
                        )

            if existing_info != new_info:
                _write_info(new_info, path)
                _info_cache[_info_hash_key(_to_info_path(path))] = new_info
                return True

        return False

    def set_voxel_offset(self, voxel_offset_and_res: tuple[Vec3D[int], Vec3D[float]]) -> None:
        voxel_offset, res = voxel_offset_and_res
        assert self.info_spec_params is not None
        self.info_spec_params.bbox = self.info_spec_params.bbox.with_start(voxel_offset, res)

    def set_chunk_size(self, chunk_size_and_res: tuple[Vec3D[int], Vec3D]) -> None:
        chunk_size, _ = chunk_size_and_res
        assert self.info_spec_params is not None
        self.info_spec_params.chunk_size = chunk_size

    def set_dataset_size(self, dataset_size_and_res: tuple[Vec3D[int], Vec3D]) -> None:
        dataset_size, res = dataset_size_and_res
        assert self.info_spec_params is not None
        self.info_spec_params.bbox = self.info_spec_params.bbox.with_end(
            self.info_spec_params.bbox.start / res + dataset_size, res
        )


@cachetools.cached(_info_cache, key=_info_hash_key)
def _get_info_from_info_path(info_path: str) -> dict[str, Any]:
    cf = CloudFile(info_path)
    if cf.exists():
        result = CloudVolume(info_path[: -len("/info")]).info
        assert isinstance(result, dict)
    else:
        raise FileNotFoundError(f"The infofile at '{info_path}' does not exist.")

    return result


def _write_info(info: dict[str, Any], path: str) -> None:  # pragma: no cover
    info_path = _to_info_path(path)
    CloudFile(info_path).put_json(info, cache_control="no-cache")


def _to_info_path(path: str) -> str:
    if not path.endswith("/info"):
        path = os.path.join(path, "info")
    return abspath(path)


def _str(n: float) -> str:  # pragma: no cover
    if int(n) == n:
        return str(int(n))
    return str(n)


def res_to_key(resolution: Sequence[float]) -> str:
    return "_".join([_str(v) for v in resolution])


def is_integer_within_eps(value: float, eps: float = 1e-5) -> bool:
    return abs(value - round(value)) < eps


def _merge_and_sort_scales(
    existing_scales: Sequence[dict[str, Any]], new_scales: Sequence[dict[str, Any]]
) -> Sequence[dict[str, Any]]:
    """Merge two list of scales, overwriting the old with new entries"""
    ret_dict = {k["key"]: k for k in list(existing_scales) + list(new_scales)}
    # sort scales by voxel volumes
    ret = [
        ret_dict[k]
        for k in sorted(ret_dict.keys(), key=lambda x: math.prod(ret_dict[x]["resolution"]))
    ]
    return ret

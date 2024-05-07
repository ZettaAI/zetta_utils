# pylint: disable=missing-docstring
from __future__ import annotations

import math
import os
from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import attrs
import cachetools
from cachetools.keys import hashkey
from cloudfiles import CloudFile

from zetta_utils.common import abspath, is_local
from zetta_utils.geometry import Vec3D

InfoExistsModes = Literal["expect_same", "overwrite"]

_info_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=500)
_info_hash_key = hashkey


# wrapper to cache using absolute paths with '/info'.
# invalidates the cached infofile if the infofile is local and has since been deleted.
def get_info(path: str) -> Dict[str, Any]:
    info_path = _to_info_path(path)
    if is_local(info_path):
        if not CloudFile(info_path).exists():
            _info_cache.pop(_info_hash_key(info_path), None)
    return _get_info_from_info_path(info_path)


@cachetools.cached(_info_cache, key=_info_hash_key)
def _get_info_from_info_path(info_path: str) -> Dict[str, Any]:
    cf = CloudFile(info_path)
    if cf.exists():
        result = CloudFile(info_path).get_json()
    else:
        raise FileNotFoundError(f"The infofile at '{info_path}' does not exist.")

    return result


def _write_info(
    info: Dict[str, Any], path: str
) -> None:  # pylint: disable=too-many-branches, consider-iterating-dictionary
    info_path = _to_info_path(path)
    CloudFile(info_path).put_json(info)


def _to_info_path(path: str) -> str:
    if not path.endswith("/info"):
        path = os.path.join(path, "info")
    return abspath(path)


def _str(n: float) -> str:  # pragma: no cover
    if int(n) == n:
        return str(int(n))
    return str(n)


def res_to_key(resolution: Vec3D) -> str:  # pragma: no cover
    return "_".join([_str(v) for v in resolution])


def _check_seq_is_int(seq: Sequence[int | float]) -> bool:
    return not all(float(k).is_integer() for k in seq)


def _get_ref_scale(
    add_scales_ref: str | dict[str, Any] | None, reference_info: dict[str, Any]
) -> dict[str, Any]:
    # the reference scale can either be a dictionary or an existing key in `ref_info`
    if isinstance(add_scales_ref, dict):
        return add_scales_ref
    # if not, search for the given key in ref_info
    if "scales" not in reference_info:
        raise RuntimeError("`scales` must be in `reference_info` if `add_scales_ref` is a key")
    if add_scales_ref is None:
        # get the highest res scale by sorting
        add_scales_ref = _merge_and_sort_scales(reference_info["scales"], [])[0]["key"]
    matched = list(filter(lambda x: x["key"] == add_scales_ref, reference_info["scales"]))
    if len(matched) == 0:
        raise RuntimeError(f'`reference_info` does not have scale "{add_scales_ref}"')
    return matched[0]


def _make_scale(ref: dict[str, Any], target: Sequence[float] | dict[str, Any]) -> dict[str, Any]:
    """Make a single scale based on the reference scale"""
    ret = {}
    if isinstance(target, dict):
        ret = target.copy()
    else:
        ret["resolution"] = target
    multiplier = [k / v for k, v in zip(ret["resolution"], ref["resolution"])]

    # fill missing values if necessary
    expected_key = res_to_key(ret["resolution"])
    if "key" not in ret:
        ret["key"] = expected_key
    else:
        # check that user provided `key` is equal to res "x_y_z"
        if ret["key"] != expected_key:
            raise RuntimeError(
                f"Scale key of {ret} has to be {expected_key} to match scale resolution"
                f" {ret['resolution']}"
            )
    if "size" not in ret:
        ret["size"] = [k / m for k, m in zip(ref["size"], multiplier)]
    if "voxel_offset" not in ret:
        ret["voxel_offset"] = [k / m for k, m in zip(ref["voxel_offset"], multiplier)]
    for k in ref:
        if k not in ret:
            ret[k] = ref[k]

    # check and convert values to int
    errored = _check_seq_is_int(ret["size"])
    errored |= _check_seq_is_int(ret["voxel_offset"])
    if errored:
        raise RuntimeError(f"Computed scale {ret} does not have integer size and offsets")
    ret["voxel_offset"] = [int(k) for k in ret["voxel_offset"]]
    ret["size"] = [int(k) for k in ret["size"]]
    return ret


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


@attrs.mutable
class PrecomputedInfoSpec:
    reference_path: str | None = None
    field_overrides: dict[str, Any] | None = None
    default_chunk_size: Sequence[int] | None = None
    default_voxel_offset: Sequence[int] | None = None
    default_dataset_size: Sequence[int] | None = None
    chunk_size_map: dict[str, Sequence[int]] | None = None
    voxel_offset_map: dict[str, Sequence[int]] | None = None
    dataset_size_map: dict[str, Sequence[int]] | None = None
    data_type: str | None = None
    add_scales: Sequence[Sequence[float] | dict[str, Any]] | None = None
    add_scales_ref: str | dict[str, Any] | None = None
    add_scales_mode: str = "merge"
    # ensure_scales: Optional[Iterable[int]] = None

    def set_voxel_offset(self, voxel_offset_and_res: Tuple[Vec3D[int], Vec3D]) -> None:
        voxel_offset, resolution = voxel_offset_and_res
        key = res_to_key(resolution)
        if self.voxel_offset_map is None:
            self.voxel_offset_map = {}

        self.voxel_offset_map[key] = voxel_offset

    def set_chunk_size(self, chunk_size_and_res: Tuple[Vec3D[int], Vec3D]) -> None:
        chunk_size, resolution = chunk_size_and_res
        key = res_to_key(resolution)
        if self.chunk_size_map is None:
            self.chunk_size_map = {}
        self.chunk_size_map[key] = chunk_size

    def set_dataset_size(self, dataset_size_and_res: Tuple[Vec3D[int], Vec3D]) -> None:
        dataset_size, resolution = dataset_size_and_res
        key = res_to_key(resolution)
        if self.dataset_size_map is None:
            self.dataset_size_map = {}
        self.dataset_size_map[key] = dataset_size

    def make_info(  # pylint: disable=too-many-branches, consider-iterating-dictionary
        self,
    ) -> Optional[Dict[str, Any]]:
        if (
            self.reference_path is None
            and self.field_overrides is None
            and self.add_scales is None
        ):
            result = None
        else:
            field_overrides = self.field_overrides
            if field_overrides is None:
                field_overrides = {}
            reference_info = {}  # type: Dict[str, Any]
            if self.reference_path is not None:
                reference_info = get_info(self.reference_path)

            result = deepcopy(reference_info)

            if self.add_scales is not None:
                if "scales" in field_overrides:
                    raise RuntimeError(
                        "`scales` must not be in `field_overrides` if `add_scales` is used"
                    )
                ref_scale = _get_ref_scale(self.add_scales_ref, reference_info)
                new_scales = [_make_scale(ref=ref_scale, target=e) for e in self.add_scales]
                if self.add_scales_mode == "replace" or "scales" not in result:
                    result["scales"] = _merge_and_sort_scales([], new_scales)
                elif self.add_scales_mode == "merge":
                    result["scales"] = _merge_and_sort_scales(result["scales"], new_scales)
                else:
                    raise RuntimeError(f"Unknown `add_scales_mode` {self.add_scales_mode}")

            result.update(field_overrides)

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
            if self.default_dataset_size is not None:
                for e in result["scales"]:
                    e["size"] = [*self.default_dataset_size]
            if self.dataset_size_map is not None:
                for e in result["scales"]:
                    if e["key"] in self.dataset_size_map.keys():
                        e["size"] = [*self.dataset_size_map[e["key"]]]
            if self.data_type is not None:
                result["data_type"] = self.data_type

            # if self.ensure_scales is not None:  # pragma: no cover
            #    raise NotImplementedError()

        return result

    """
    Update infofile at `path`. Returns True if there was a write, else returns False.
    """

    def update_info(self, path: str, on_info_exists: InfoExistsModes) -> bool:
        try:
            existing_info = get_info(path)
        except FileNotFoundError:
            existing_info = None
        new_info = self.make_info()
        self.reference_path = path
        self.add_scales = None  # no need to and cannot add scales after ref path is updated
        if new_info is None and existing_info is None:
            raise RuntimeError(  # pragma: no cover
                f"The infofile at {path} does not exist, but the infospec "
                f"given is empty with no reference path."
            )

        if new_info is not None:
            if (
                existing_info is not None
                and on_info_exists == "expect_same"
                and new_info != existing_info
            ):
                raise RuntimeError(
                    f"Info created by the info_spec {self} is not equal to "
                    f"info existing at '{path}' "
                    "while `on_info_exists` is set to 'expect_same'"
                )
            if existing_info != new_info:
                _write_info(new_info, path)
                _info_cache[_info_hash_key(_to_info_path(path))] = new_info
                return True

        return False

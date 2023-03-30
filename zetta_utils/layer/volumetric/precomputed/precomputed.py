# pylint: disable=missing-docstring
from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import attrs
import cachetools
import fsspec
import fsspec.asyn
from cachetools.keys import hashkey

from zetta_utils.common import abspath
from zetta_utils.geometry import Vec3D

InfoExistsModes = Literal["expect_same", "overwrite"]

_info_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=500)
_info_hash_key = hashkey


@cachetools.cached(_info_cache, key=_info_hash_key)
def get_info(path: str) -> Dict[str, Any]:
    path = abspath(path)
    if not path.endswith("/info"):
        path = os.path.join(path, "info")
    try:
        fsspec.asyn.reset_lock()  # https://github.com/fsspec/gcsfs/issues/379
        with fsspec.open(path) as f:
            result = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"'{path}' does not have an infofile.") from e
    return result


def _write_info(
    info: Dict[str, Any], path: str
) -> None:  # pylint: disable=too-many-branches, consider-iterating-dictionary
    path = abspath(path)
    if not path.endswith("/info"):
        path = os.path.join(path, "info")
    fsspec.asyn.reset_lock()  # https://github.com/fsspec/gcsfs/issues/379
    with fsspec.open(path, "w") as f:
        json.dump(info, f)


def _str(n: float) -> str:  # pragma: no cover
    if int(n) == n:
        return str(int(n))
    return str(n)


@attrs.mutable
class PrecomputedInfoSpec:
    reference_path: str | None = None
    field_overrides: dict[str, Any] | None = None
    default_chunk_size: Sequence[int] | None = None
    default_voxel_offset: Sequence[int] | None = None
    chunk_size_map: dict[str, Sequence[int]] | None = None
    voxel_offset_map: dict[str, Sequence[int]] | None = None
    data_type: str | None = None
    # ensure_scales: Optional[Iterable[int]] = None

    def set_voxel_offset(self, voxel_offset_and_res: Tuple[Vec3D[int], Vec3D]) -> None:
        voxel_offset, resolution = voxel_offset_and_res
        key = "_".join([_str(v) for v in resolution])
        if self.voxel_offset_map is None:
            self.voxel_offset_map = {}

        self.voxel_offset_map[key] = voxel_offset

    def set_chunk_size(self, chunk_size_and_res: Tuple[Vec3D[int], Vec3D]) -> None:
        chunk_size, resolution = chunk_size_and_res
        key = "_".join([_str(v) for v in resolution])
        if self.chunk_size_map is None:
            self.chunk_size_map = {}
        self.chunk_size_map[key] = chunk_size

    def make_info(  # pylint: disable=too-many-branches, consider-iterating-dictionary
        self,
    ) -> Optional[Dict[str, Any]]:
        if self.reference_path is None and self.field_overrides is None:
            result = None
        else:
            field_overrides = self.field_overrides
            if field_overrides is None:
                field_overrides = {}
            reference_info = {}  # type: Dict[str, Any]
            if self.reference_path is not None:
                reference_info = get_info(self.reference_path)
            result = deepcopy({**reference_info, **field_overrides})
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
                _info_cache[_info_hash_key(path)] = new_info
                return True

        return False

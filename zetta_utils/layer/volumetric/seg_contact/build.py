from __future__ import annotations

from typing import Literal

import fsspec

from zetta_utils.builder import register

from .backend import SegContactLayerBackend
from .info_spec import SegContactInfoSpec
from .layer import SegContactDataProcT, VolumetricSegContactLayer

register("SegContactLayerBackend.from_path")(SegContactLayerBackend.from_path)


@register("build_seg_contact_layer")
def build_seg_contact_layer(
    path: str,
    readonly: bool = False,
    info_spec: SegContactInfoSpec | None = None,
    mode: Literal["read", "write", "update"] = "read",
    info_overwrite: bool = False,
    info_keep_existing_pointcloud_configs: bool = True,
    read_procs: tuple[SegContactDataProcT, ...] = (),
    local_point_clouds: list[tuple[int, int]] | None = None,
) -> VolumetricSegContactLayer:
    """Build a VolumetricSegContactLayer from a path.

    :param path: Path to seg_contact layer.
    :param readonly: Whether the layer should be read-only.
    :param info_spec: Info specification for creating new layer.
    :param mode: How the layer should be opened:
        - "read": for reading only; layer must exist.
        - "write": for writing; creates new layer, fails if exists.
        - "update": for writing to existing layer.
    :param info_overwrite: Whether to allow overwriting existing info fields.
    :param info_keep_existing_pointcloud_configs: Whether to keep existing pointcloud
        configs when updating info (merge new with existing).
    :param read_procs: Tuple of data processors to apply on read.
    :param local_point_clouds: List of (radius_nm, n_points) tuples specifying which
        pointcloud configs to load. If None, loads all configs from info file.
    :return: VolumetricSegContactLayer instance.
    """
    info_path = f"{path}/info"
    fs, fs_path = fsspec.core.url_to_fs(info_path)
    layer_exists = fs.exists(fs_path)

    if mode == "read":
        if not layer_exists:
            raise FileNotFoundError(f"SegContact layer not found at {path}")
        backend = SegContactLayerBackend.from_path(path)
        if local_point_clouds is not None:
            backend = backend.with_changes(local_point_clouds=local_point_clouds)
        return VolumetricSegContactLayer(backend=backend, readonly=True, read_procs=read_procs)

    if mode == "write":
        if info_spec is None:
            raise ValueError("info_spec is required when mode='write'")
        if layer_exists and not info_overwrite:
            raise RuntimeError(f"Layer already exists at {path} and info_overwrite=False")
        info_spec.update_info(
            path,
            overwrite=info_overwrite,
            keep_existing_pointcloud_configs=info_keep_existing_pointcloud_configs,
        )
        backend = SegContactLayerBackend.from_path(path)
        if local_point_clouds is not None:
            backend = backend.with_changes(local_point_clouds=local_point_clouds)
        return VolumetricSegContactLayer(backend=backend, readonly=readonly, read_procs=read_procs)

    if mode == "update":
        if not layer_exists:
            raise FileNotFoundError(f"SegContact layer not found at {path}")
        if info_spec is not None:
            info_spec.update_info(
                path,
                overwrite=info_overwrite,
                keep_existing_pointcloud_configs=info_keep_existing_pointcloud_configs,
            )
        backend = SegContactLayerBackend.from_path(path)
        if local_point_clouds is not None:
            backend = backend.with_changes(local_point_clouds=local_point_clouds)
        return VolumetricSegContactLayer(backend=backend, readonly=readonly, read_procs=read_procs)

    raise ValueError(f"Invalid mode: {mode}")

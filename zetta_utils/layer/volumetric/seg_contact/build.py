from __future__ import annotations

from typing import Literal

from zetta_utils.builder import register

from .backend import SegContactLayerBackend
from .info_spec import SegContactInfoSpec
from .layer import VolumetricSegContactLayer


@register("build_seg_contact_layer")
def build_seg_contact_layer(
    path: str,
    readonly: bool = False,
    info_spec: SegContactInfoSpec | None = None,
    mode: Literal["read", "write", "update"] = "read",
) -> VolumetricSegContactLayer:
    """Build a VolumetricSegContactLayer from a path.

    :param path: Path to seg_contact layer.
    :param readonly: Whether the layer should be read-only.
    :param info_spec: Info specification for creating new layer.
    :param mode: How the layer should be opened:
        - "read": for reading only; layer must exist.
        - "write": for writing; creates new layer, fails if exists.
        - "update": for writing to existing layer.
    :return: VolumetricSegContactLayer instance.
    """
    import os

    info_path = os.path.join(path, "info")
    layer_exists = os.path.exists(info_path)

    if mode == "read":
        if not layer_exists:
            raise FileNotFoundError(f"SegContact layer not found at {path}")
        backend = SegContactLayerBackend.from_path(path)
        return VolumetricSegContactLayer(backend=backend, readonly=True)

    if mode == "write":
        if layer_exists:
            raise FileExistsError(f"SegContact layer already exists at {path}")
        if info_spec is None:
            raise ValueError("info_spec is required when mode='write'")
        info_spec.write_info(path)
        backend = SegContactLayerBackend.from_path(path)
        return VolumetricSegContactLayer(backend=backend, readonly=readonly)

    if mode == "update":
        if not layer_exists:
            raise FileNotFoundError(f"SegContact layer not found at {path}")
        backend = SegContactLayerBackend.from_path(path)
        return VolumetricSegContactLayer(backend=backend, readonly=readonly)

    raise ValueError(f"Invalid mode: {mode}")

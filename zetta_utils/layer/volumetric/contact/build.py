from __future__ import annotations

from zetta_utils.builder import register

from .backend import ContactLayerBackend
from .layer import VolumetricContactLayer


@register("build_contact_layer")
def build_contact_layer(
    path: str,
    readonly: bool = False,
) -> VolumetricContactLayer:
    """Build a VolumetricContactLayer from a path."""
    raise NotImplementedError

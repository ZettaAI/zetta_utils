from __future__ import annotations

from typing import Sequence

import attrs
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.geometry import BBox3D, Vec3D


@typechecked
@attrs.mutable
class SegContactInfoSpecParams:
    """Parameters for creating a seg_contact layer info file."""

    resolution: Vec3D[int]
    chunk_size: Vec3D[int]
    max_contact_span: int
    bbox: BBox3D

    @classmethod
    def from_reference(
        cls,
        reference_path: str,
        resolution: Sequence[int] | None = None,
        chunk_size: Sequence[int] | None = None,
        max_contact_span: int | None = None,
        bbox: BBox3D | None = None,
    ) -> SegContactInfoSpecParams:
        """Create params from a reference seg_contact layer path."""
        from .backend import SegContactLayerBackend

        ref = SegContactLayerBackend.from_path(reference_path)

        if resolution is None:
            resolution = ref.resolution
        if chunk_size is None:
            chunk_size = ref.chunk_size
        if max_contact_span is None:
            max_contact_span = ref.max_contact_span
        if bbox is None:
            bbox = BBox3D.from_coords(
                start_coord=ref.voxel_offset,
                end_coord=Vec3D(*ref.voxel_offset) + Vec3D(*ref.size),
                resolution=ref.resolution,
            )

        return cls(
            resolution=Vec3D(*resolution),
            chunk_size=Vec3D(*chunk_size),
            max_contact_span=max_contact_span,
            bbox=bbox,
        )


@typechecked
@attrs.mutable
class SegContactInfoSpec:
    """Specification for seg_contact layer info file, similar to PrecomputedInfoSpec."""

    info_path: str | None = None
    info_spec_params: SegContactInfoSpecParams | None = None

    def __attrs_post_init__(self):
        if (self.info_path is None and self.info_spec_params is None) or (
            self.info_path is not None and self.info_spec_params is not None
        ):
            raise ValueError("Exactly one of `info_path`/`info_spec_params` must be provided")

    def make_info(self) -> dict:
        """Generate info dict from spec params."""
        if self.info_path is not None:
            from .backend import SegContactLayerBackend

            backend = SegContactLayerBackend.from_path(self.info_path)
            return {
                "format_version": "1.0",
                "type": "seg_contact",
                "resolution": list(backend.resolution),
                "voxel_offset": list(backend.voxel_offset),
                "size": list(backend.size),
                "chunk_size": list(backend.chunk_size),
                "max_contact_span": backend.max_contact_span,
            }
        else:
            assert self.info_spec_params is not None
            params = self.info_spec_params
            voxel_offset = [int(params.bbox.start[i] / params.resolution[i]) for i in range(3)]
            size = [int(params.bbox.shape[i] / params.resolution[i]) for i in range(3)]
            return {
                "format_version": "1.0",
                "type": "seg_contact",
                "resolution": list(params.resolution),
                "voxel_offset": voxel_offset,
                "size": size,
                "chunk_size": list(params.chunk_size),
                "max_contact_span": params.max_contact_span,
            }

    def write_info(self, path: str) -> None:
        """Write info file to the given path."""
        import json
        import os

        info = self.make_info()
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "info"), "w") as f:
            json.dump(info, f, indent=2)

    def set_bbox(self, bbox: BBox3D) -> None:
        """Update the bounding box."""
        assert self.info_spec_params is not None
        self.info_spec_params.bbox = bbox


@builder.register("build_seg_contact_info_spec")
def build_seg_contact_info_spec(
    info_path: str | None = None,
    reference_path: str | None = None,
    resolution: Sequence[int] | None = None,
    chunk_size: Sequence[int] | None = None,
    max_contact_span: int | None = None,
    bbox: BBox3D | None = None,
) -> SegContactInfoSpec:
    """Build a SegContactInfoSpec for use in specs.

    :param info_path: Path to existing seg_contact layer to use as info source.
    :param reference_path: Path to reference seg_contact layer to inherit params from.
    :param resolution: Voxel resolution in nm (x, y, z).
    :param chunk_size: Chunk size in voxels (x, y, z).
    :param max_contact_span: Maximum contact span in voxels.
    :param bbox: Bounding box for the dataset.
    :return: SegContactInfoSpec instance.
    """
    if info_path is not None:
        if any(p is not None for p in [reference_path, resolution, chunk_size, max_contact_span]):
            raise ValueError("When `info_path` is provided, other params should not be specified")
        return SegContactInfoSpec(info_path=info_path)

    if reference_path is not None:
        params = SegContactInfoSpecParams.from_reference(
            reference_path=reference_path,
            resolution=resolution,
            chunk_size=chunk_size,
            max_contact_span=max_contact_span,
            bbox=bbox,
        )
    else:
        if resolution is None or chunk_size is None or max_contact_span is None or bbox is None:
            raise ValueError(
                "When no reference is provided, resolution, chunk_size, "
                "max_contact_span, and bbox are all required"
            )
        params = SegContactInfoSpecParams(
            resolution=Vec3D(*resolution),
            chunk_size=Vec3D(*chunk_size),
            max_contact_span=max_contact_span,
            bbox=bbox,
        )

    return SegContactInfoSpec(info_spec_params=params)

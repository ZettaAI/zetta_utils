from __future__ import annotations

import json
from typing import Sequence

import attrs
from cloudfiles import CloudFile
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.geometry import BBox3D, Vec3D

from .contact import read_info


@typechecked
@attrs.mutable
class SegContactInfoSpecParams:
    """Parameters for creating a seg_contact layer info file."""

    resolution: Vec3D[int]
    chunk_size: Vec3D[int]
    max_contact_span: int
    bbox: BBox3D
    segmentation_path: str | None = None
    affinity_path: str | None = None
    image_path: str | None = None
    local_point_clouds: list[dict] | None = None
    merge_decisions: list[str] | None = None
    filter_settings: dict | None = None

    @classmethod
    def from_reference(
        cls,
        reference_path: str,
        resolution: Sequence[int] | None = None,
        chunk_size: Sequence[int] | None = None,
        max_contact_span: int | None = None,
        bbox: BBox3D | None = None,
        segmentation_path: str | None = None,
        affinity_path: str | None = None,
        image_path: str | None = None,
        local_point_clouds: list[dict] | None = None,
        merge_decisions: list[str] | None = None,
        filter_settings: dict | None = None,
    ) -> SegContactInfoSpecParams:
        """Create params from a reference seg_contact layer path."""
        ref_info = read_info(reference_path)

        if resolution is None:
            resolution = ref_info["resolution"]
        if chunk_size is None:
            chunk_size = ref_info["chunk_size"]
        if max_contact_span is None:
            max_contact_span = ref_info["max_contact_span"]
        if bbox is None:
            voxel_offset: Vec3D[int] = Vec3D(*ref_info["voxel_offset"])
            size: Vec3D[int] = Vec3D(*ref_info["size"])
            bbox = BBox3D.from_coords(
                start_coord=voxel_offset,
                end_coord=voxel_offset + size,
                resolution=Vec3D(*resolution),
            )
        if segmentation_path is None:
            segmentation_path = ref_info.get("segmentation_path")
        if affinity_path is None:
            affinity_path = ref_info.get("affinity_path")
        if image_path is None:
            image_path = ref_info.get("image_path")
        if local_point_clouds is None:
            local_point_clouds = ref_info.get("local_point_clouds")
        if merge_decisions is None:
            merge_decisions = ref_info.get("merge_decisions")
        if filter_settings is None:
            filter_settings = ref_info.get("filter_settings")

        return cls(
            resolution=Vec3D(*resolution),
            chunk_size=Vec3D(*chunk_size),
            max_contact_span=max_contact_span,
            bbox=bbox,
            segmentation_path=segmentation_path,
            affinity_path=affinity_path,
            image_path=image_path,
            local_point_clouds=local_point_clouds,
            merge_decisions=merge_decisions,
            filter_settings=filter_settings,
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
            return read_info(self.info_path)
        else:
            assert self.info_spec_params is not None
            params = self.info_spec_params
            voxel_offset = [int(params.bbox.start[i] / params.resolution[i]) for i in range(3)]
            size = [int(params.bbox.shape[i] / params.resolution[i]) for i in range(3)]
            info: dict = {
                "format_version": "1.0",
                "type": "seg_contact",
                "resolution": list(params.resolution),
                "voxel_offset": voxel_offset,
                "size": size,
                "chunk_size": list(params.chunk_size),
                "max_contact_span": params.max_contact_span,
            }
            if params.segmentation_path:
                info["segmentation_path"] = params.segmentation_path
            if params.affinity_path:
                info["affinity_path"] = params.affinity_path
            if params.image_path:
                info["image_path"] = params.image_path
            if params.local_point_clouds:
                info["local_point_clouds"] = params.local_point_clouds
            if params.merge_decisions:
                info["merge_decisions"] = params.merge_decisions
            if params.filter_settings:
                info["filter_settings"] = params.filter_settings
            return info

    def write_info(self, path: str) -> None:
        """Write info file to the given path."""
        info = self.make_info()
        info_path = f"{path}/info"
        CloudFile(info_path).put(json.dumps(info, indent=2).encode("utf-8"))

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
    segmentation_path: str | None = None,
    affinity_path: str | None = None,
    image_path: str | None = None,
    local_point_clouds: list[dict] | None = None,
    merge_decisions: list[str] | None = None,
    filter_settings: dict | None = None,
) -> SegContactInfoSpec:
    """Build a SegContactInfoSpec for use in specs.

    :param info_path: Path to existing seg_contact layer to use as info source.
    :param reference_path: Path to reference seg_contact layer to inherit params from.
    :param resolution: Voxel resolution in nm (x, y, z).
    :param chunk_size: Chunk size in voxels (x, y, z).
    :param max_contact_span: Maximum contact span in voxels.
    :param bbox: Bounding box for the dataset.
    :param segmentation_path: Path to source segmentation layer.
    :param affinity_path: Path to source affinity layer.
    :param image_path: Path to image layer for visualization.
    :param local_point_clouds: List of pointcloud configs [{radius_nm, n_points}].
    :param merge_decisions: List of merge decision authority names.
    :param filter_settings: Filter parameters used during generation.
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
            segmentation_path=segmentation_path,
            affinity_path=affinity_path,
            image_path=image_path,
            local_point_clouds=local_point_clouds,
            merge_decisions=merge_decisions,
            filter_settings=filter_settings,
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
            segmentation_path=segmentation_path,
            affinity_path=affinity_path,
            image_path=image_path,
            local_point_clouds=local_point_clouds,
            merge_decisions=merge_decisions,
            filter_settings=filter_settings,
        )

    return SegContactInfoSpec(info_spec_params=params)

from __future__ import annotations

from typing import Sequence

import attrs

from zetta_utils import builder
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.seg_contact import read_info

from .base import SampleIndexer


@builder.register("SegContactIndexer")
@attrs.frozen
class SegContactIndexer(SampleIndexer[VolumetricIndex]):
    """SampleIndexer for seg_contact data - returns chunk bounding boxes.

    Computes chunk grid from info file (voxel_offset, size, chunk_size) and returns
    VolumetricIndex for each chunk, intersected with the optional bbox filter.

    :param path: Path to seg_contact layer (contains info file).
    :param bbox: Optional bounding box to restrict sampling. If None, uses full layer bounds.
    :param resolution: Resolution for the bounding box. If None, uses layer resolution.
    """

    path: str
    bbox: BBox3D | None = None
    resolution: Sequence[float] | None = None

    _chunk_bounds: list[tuple[int, int, int, int, int, int]] = attrs.field(
        init=False, factory=list
    )
    _resolution: Vec3D[int] = attrs.field(init=False)
    _filter_bbox: BBox3D | None = attrs.field(init=False)

    def __attrs_post_init__(self):
        # Read info to get resolution, size, offset, chunk_size
        info = read_info(self.path)
        resolution: Vec3D[int] = Vec3D(*info["resolution"])
        voxel_offset = info["voxel_offset"]
        size = info["size"]
        chunk_size = info["chunk_size"]
        object.__setattr__(self, "_resolution", resolution)

        # Determine filter bbox in nm
        if self.bbox is not None:
            filter_bbox = self.bbox
        else:
            filter_bbox = None
        object.__setattr__(self, "_filter_bbox", filter_bbox)

        # Compute chunk grid from info
        chunk_bounds = []
        for x in range(voxel_offset[0], voxel_offset[0] + size[0], chunk_size[0]):
            for y in range(voxel_offset[1], voxel_offset[1] + size[1], chunk_size[1]):
                for z in range(voxel_offset[2], voxel_offset[2] + size[2], chunk_size[2]):
                    x_end = min(x + chunk_size[0], voxel_offset[0] + size[0])
                    y_end = min(y + chunk_size[1], voxel_offset[1] + size[1])
                    z_end = min(z + chunk_size[2], voxel_offset[2] + size[2])
                    bounds = (x, x_end, y, y_end, z, z_end)

                    # Check if chunk overlaps with filter bbox
                    if filter_bbox is not None:
                        chunk_bbox = self._bounds_to_bbox(bounds)
                        if not self._bboxes_overlap(chunk_bbox, filter_bbox):
                            continue
                    chunk_bounds.append(bounds)

        # Sort for deterministic ordering
        chunk_bounds.sort()
        object.__setattr__(self, "_chunk_bounds", chunk_bounds)

    def _bounds_to_bbox(self, bounds: tuple[int, int, int, int, int, int]) -> BBox3D:
        """Convert voxel bounds to BBox3D in nm."""
        x_start, x_end, y_start, y_end, z_start, z_end = bounds
        return BBox3D.from_coords(
            start_coord=[x_start, y_start, z_start],
            end_coord=[x_end, y_end, z_end],
            resolution=list(self._resolution),
        )

    def _bboxes_overlap(self, a: BBox3D, b: BBox3D) -> bool:
        """Check if two bboxes overlap."""
        return not (
            a.end[0] <= b.start[0]
            or b.end[0] <= a.start[0]
            or a.end[1] <= b.start[1]
            or b.end[1] <= a.start[1]
            or a.end[2] <= b.start[2]
            or b.end[2] <= a.start[2]
        )

    def __len__(self) -> int:
        return len(self._chunk_bounds)

    def __call__(self, idx: int) -> VolumetricIndex:
        """Get VolumetricIndex for chunk at given index.

        Returns the intersection of chunk bounds with filter bbox (if any).

        :param idx: Chunk index (0 to len-1).
        :return: VolumetricIndex for that chunk region.
        """
        if idx < 0 or idx >= len(self._chunk_bounds):
            raise IndexError(f"Index {idx} out of range [0, {len(self._chunk_bounds)})")

        bounds = self._chunk_bounds[idx]
        chunk_bbox = self._bounds_to_bbox(bounds)

        # Intersect with filter bbox if provided
        if self._filter_bbox is not None:
            chunk_bbox = chunk_bbox.intersection(self._filter_bbox)

        resolution: Vec3D = Vec3D(*(self.resolution or self._resolution))
        return VolumetricIndex(bbox=chunk_bbox, resolution=resolution)

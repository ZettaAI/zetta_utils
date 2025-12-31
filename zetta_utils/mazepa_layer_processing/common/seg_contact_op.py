from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import attrs
import cc3d
import numpy as np
import pandas as pd
import trimesh
from cloudvolume import CloudVolume

from zetta_utils import builder
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.layer.volumetric.seg_contact import SegContact, VolumetricSegContactLayer
from zetta_utils.mazepa import taskable_operation_cls
from zetta_utils.mazepa.semaphores import semaphore


def _read_layers_parallel(
    segmentation: VolumetricLayer,
    reference: VolumetricLayer,
    affinity: VolumetricLayer,
    idx: VolumetricIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read segmentation, reference, and affinity layers in parallel."""

    def read_layer(layer: VolumetricLayer, index: VolumetricIndex) -> np.ndarray:
        return np.asarray(layer[index])

    with ThreadPoolExecutor(max_workers=3) as executor:
        seg_future = executor.submit(read_layer, segmentation, idx)
        ref_future = executor.submit(read_layer, reference, idx)
        aff_future = executor.submit(read_layer, affinity, idx)
        return seg_future.result().squeeze(), ref_future.result().squeeze(), aff_future.result()


def _compute_overlaps(
    seg: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute overlaps between segments and reference connected components."""
    cc_ref = cc3d.connected_components(reference, connectivity=6)
    flat_seg = seg.ravel()
    flat_cc_ref = cc_ref.ravel()
    valid_mask = (flat_seg != 0) & (flat_cc_ref != 0)
    df = pd.DataFrame({"seg": flat_seg[valid_mask], "cc_ref": flat_cc_ref[valid_mask]})
    counts_df = df.groupby(["seg", "cc_ref"]).size().reset_index(name="count")
    return (
        counts_df["seg"].values.astype(np.int64),
        counts_df["cc_ref"].values.astype(np.int64),
        counts_df["count"].values.astype(np.int32),
    )


def _find_small_segment_ids(seg: np.ndarray, min_seg_size_vx: int) -> set[int]:
    """Find segment IDs with total voxel count below threshold."""
    unique, counts = np.unique(seg, return_counts=True)
    return {int(s) for s, cnt in zip(unique, counts) if s != 0 and cnt < min_seg_size_vx}


def _find_merger_segment_ids(
    seg_ids: np.ndarray, ref_ids: np.ndarray, counts: np.ndarray, min_overlap_vx: int
) -> set[int]:
    """Find merger segments (overlap 2+ reference CCs with >= min_overlap each)."""
    seg_to_ref: dict[int, set[int]] = defaultdict(set)
    for seg, ref, cnt in zip(seg_ids, ref_ids, counts):
        if cnt >= min_overlap_vx:
            seg_to_ref[int(seg)].add(int(ref))
    return {seg for seg, refs in seg_to_ref.items() if len(refs) >= 2}


def _find_unclaimed_segment_ids(
    seg_ids: np.ndarray, counts: np.ndarray, min_overlap_vx: int
) -> set[int]:
    """Find segments without sufficient reference overlap."""
    seg_max_overlap: dict[int, int] = defaultdict(int)
    for seg, cnt in zip(seg_ids, counts):
        seg_max_overlap[int(seg)] = max(seg_max_overlap[int(seg)], int(cnt))
    return {seg for seg, max_ovl in seg_max_overlap.items() if max_ovl < min_overlap_vx}


def _build_seg_to_ref(
    seg_ids: np.ndarray, ref_ids: np.ndarray, counts: np.ndarray, min_overlap_vx: int
) -> dict[int, set[int]]:
    """Build mapping from segment to reference CCs it overlaps with."""
    result: dict[int, set[int]] = defaultdict(set)
    for seg, ref, cnt in zip(seg_ids, ref_ids, counts):
        if cnt >= min_overlap_vx:
            result[int(seg)].add(int(ref))
    return result


def _blackout_segments(seg: np.ndarray, ids_to_remove: set[int]) -> np.ndarray:
    """Set specified segment IDs to 0."""
    if not ids_to_remove:
        return seg
    seg = seg.copy()
    mask = np.isin(seg, list(ids_to_remove))
    seg[mask] = 0
    return seg


def _find_axis_contacts(
    seg_lo: np.ndarray,
    seg_hi: np.ndarray,
    aff_slice: np.ndarray,
    offset: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find contacts along one axis. Returns face centers."""
    mask = (seg_lo != seg_hi) & (seg_lo != 0) & (seg_hi != 0)
    idx = np.nonzero(mask)
    if len(idx[0]) == 0:
        empty_i, empty_f = np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        return empty_i, empty_i, empty_f, empty_f, empty_f, empty_f
    return (
        seg_lo[mask],
        seg_hi[mask],
        aff_slice[mask],
        idx[0].astype(np.float32) + offset[0],
        idx[1].astype(np.float32) + offset[1],
        idx[2].astype(np.float32) + offset[2],
    )


def _find_contacts(
    seg: np.ndarray, aff: np.ndarray, start: Vec3D
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find contacts between segments using affinity data."""
    sx, sy, sz = float(start[0]), float(start[1]), float(start[2])
    results = []

    for seg_lo, seg_hi, aff_slice, offset in [
        (seg[:-1], seg[1:], aff[0, 1:], (sx + 0.5, sy, sz)),
        (seg[:, :-1], seg[:, 1:], aff[1, :, 1:], (sx, sy + 0.5, sz)),
        (seg[:, :, :-1], seg[:, :, 1:], aff[2, :, :, 1:], (sx, sy, sz + 0.5)),
    ]:
        r = _find_axis_contacts(seg_lo, seg_hi, aff_slice, offset)
        if len(r[0]) > 0:
            results.append(r)

    if not results:
        empty_i, empty_f = np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        return empty_i, empty_i, empty_f, empty_f, empty_f, empty_f

    seg_a = np.concatenate([r[0] for r in results])
    seg_b = np.concatenate([r[1] for r in results])
    aff_vals = np.concatenate([r[2] for r in results])
    x = np.concatenate([r[3] for r in results])
    y = np.concatenate([r[4] for r in results])
    z = np.concatenate([r[5] for r in results])

    swap = seg_a > seg_b
    seg_a, seg_b = np.where(swap, seg_b, seg_a), np.where(swap, seg_a, seg_b)

    return seg_a, seg_b, aff_vals, x, y, z


def _filter_pairs_to_kernel(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    aff: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    start: Vec3D,
    shape: tuple[int, ...],
    crop_pad: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exclude pairs that have any contact outside the kernel region."""
    kernel_start = np.array([start[0], start[1], start[2]]) + np.array(crop_pad)
    kernel_end = np.array([start[0], start[1], start[2]]) + np.array(shape) - np.array(crop_pad)
    outside_kernel = (
        (x < kernel_start[0])
        | (x >= kernel_end[0])
        | (y < kernel_start[1])
        | (y >= kernel_end[1])
        | (z < kernel_start[2])
        | (z >= kernel_end[2])
    )
    pairs_outside: set[tuple[int, int]] = set()
    for a, b, out in zip(seg_a, seg_b, outside_kernel):
        if out:
            pairs_outside.add((int(a), int(b)))
    keep = np.array([(int(a), int(b)) not in pairs_outside for a, b in zip(seg_a, seg_b)])
    return seg_a[keep], seg_b[keep], aff[keep], x[keep], y[keep], z[keep]


def _compute_contact_counts(seg_a: np.ndarray, seg_b: np.ndarray) -> dict[tuple[int, int], int]:
    """Count contacts per segment pair."""
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for a, b in zip(seg_a, seg_b):
        counts[(int(a), int(b))] += 1
    return counts


def _build_contact_lookup(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    cz: np.ndarray,
    aff: np.ndarray,
) -> dict[tuple[int, int], list[tuple[float, float, float, float]]]:
    """Build lookup from segment pair to contact face centers."""
    data: dict[tuple[int, int], list[tuple[float, float, float, float]]] = defaultdict(list)
    for a, b, x, y, z, af in zip(seg_a, seg_b, cx, cy, cz, aff):
        data[(int(a), int(b))].append((float(x), float(y), float(z), float(af)))
    return data


def _download_and_clip_meshes(
    cv: CloudVolume, segment_ids: list[int], bbox_start: np.ndarray, bbox_end: np.ndarray
) -> dict[int, trimesh.Trimesh]:
    """Download meshes and clip to bounding box."""
    if not segment_ids:
        return {}

    meshes = cv.mesh.get(segment_ids, progress=False)
    box = trimesh.creation.box(extents=bbox_end - bbox_start)
    box.apply_translation((bbox_start + bbox_end) / 2)

    result: dict[int, trimesh.Trimesh] = {}
    for seg_id in segment_ids:
        mesh_obj = meshes.get(seg_id)
        if mesh_obj is None or len(mesh_obj.vertices) == 0 or len(mesh_obj.faces) == 0:
            continue
        mesh = trimesh.Trimesh(vertices=mesh_obj.vertices, faces=mesh_obj.faces)
        clipped = mesh.slice_plane(box.facets_origin, -np.array(box.facets_normal))
        if clipped is not None and len(clipped.vertices) > 0 and len(clipped.faces) > 0:
            result[seg_id] = clipped
    return result


def _crop_mesh_to_sphere(
    mesh: trimesh.Trimesh, center: np.ndarray, radius: float
) -> trimesh.Trimesh | None:
    """Clip mesh to sphere, keeping only faces fully inside."""
    vertex_dists = np.linalg.norm(mesh.vertices - center, axis=1)
    vertex_inside = vertex_dists <= radius
    face_inside = vertex_inside[mesh.faces].all(axis=1)
    if not face_inside.any():
        return None
    result = mesh.submesh([face_inside], append=True)
    if isinstance(result, list):
        return result[0] if result else None
    return result


def _sample_mesh_points(mesh: trimesh.Trimesh | None, n: int) -> np.ndarray:
    """Sample N points from mesh surface, area-weighted."""
    if mesh is None or len(mesh.faces) == 0:
        return np.zeros((n, 3), dtype=np.float32)
    result = trimesh.sample.sample_surface(mesh, n)
    points = result[0]
    return points.astype(np.float32)


def _compute_affinity_weighted_com(
    contacts: list[tuple[float, float, float, float]], resolution: np.ndarray
) -> np.ndarray:
    """Compute affinity-weighted center of mass in nm."""
    x = np.array([c[0] for c in contacts])
    y = np.array([c[1] for c in contacts])
    z = np.array([c[2] for c in contacts])
    aff = np.array([c[3] for c in contacts])
    aff_sum = aff.sum()
    if aff_sum == 0:
        return np.array(
            [x.mean() * resolution[0], y.mean() * resolution[1], z.mean() * resolution[2]]
        )
    return np.array(
        [
            (x * aff).sum() / aff_sum * resolution[0],
            (y * aff).sum() / aff_sum * resolution[1],
            (z * aff).sum() / aff_sum * resolution[2],
        ]
    )


def _all_contacts_in_sphere(
    contacts: list[tuple[float, float, float, float]],
    center: np.ndarray,
    radius: float,
    resolution: np.ndarray,
) -> bool:
    """Check if all contacts are within sphere radius of center."""
    for x, y, z, _ in contacts:
        pos_nm = np.array([x * resolution[0], y * resolution[1], z * resolution[2]])
        if np.linalg.norm(pos_nm - center) > radius:
            return False
    return True


def _make_contact_faces_array(
    contacts: list[tuple[float, float, float, float]], resolution: np.ndarray
) -> np.ndarray:
    """Create contact faces array (N, 4) with x, y, z, affinity in nm."""
    x = np.array([c[0] for c in contacts]) * resolution[0]
    y = np.array([c[1] for c in contacts]) * resolution[1]
    z = np.array([c[2] for c in contacts]) * resolution[2]
    aff = np.array([c[3] for c in contacts])
    return np.stack([x, y, z, aff], axis=1).astype(np.float32)


def _generate_seg_contact(
    contact_id: int,
    seg_a_id: int,
    seg_b_id: int,
    meshes: dict[int, trimesh.Trimesh],
    contact_data: dict[tuple[int, int], list[tuple[float, float, float, float]]],
    seg_to_ref: dict[int, set[int]],
    resolution: np.ndarray,
    sphere_radius_nm: float,
    n_pointcloud_points: int,
    merge_authority: str,
) -> SegContact | None:
    """Generate a single SegContact for a contact pair."""
    mesh_a, mesh_b = meshes.get(seg_a_id), meshes.get(seg_b_id)
    if mesh_a is None or mesh_b is None:
        return None

    contacts = contact_data[(seg_a_id, seg_b_id)]
    com = _compute_affinity_weighted_com(contacts, resolution)

    mesh_a_cropped = _crop_mesh_to_sphere(mesh_a, com, sphere_radius_nm)
    mesh_b_cropped = _crop_mesh_to_sphere(mesh_b, com, sphere_radius_nm)
    if mesh_a_cropped is None or mesh_b_cropped is None:
        return None

    if not _all_contacts_in_sphere(contacts, com, sphere_radius_nm, resolution):
        return None

    contact_faces = _make_contact_faces_array(contacts, resolution)

    # Sample pointclouds
    pointcloud_a = _sample_mesh_points(mesh_a_cropped, n_pointcloud_points)
    pointcloud_b = _sample_mesh_points(mesh_b_cropped, n_pointcloud_points)
    local_pointclouds = {seg_a_id: pointcloud_a, seg_b_id: pointcloud_b}

    # Compute merge decision: should merge if both segments overlap same reference CC
    should_merge = bool(seg_to_ref.get(seg_a_id, set()) & seg_to_ref.get(seg_b_id, set()))
    merge_decisions = {merge_authority: should_merge}

    return SegContact(
        id=contact_id,
        seg_a=seg_a_id,
        seg_b=seg_b_id,
        com=Vec3D(float(com[0]), float(com[1]), float(com[2])),
        contact_faces=contact_faces,
        local_pointclouds=local_pointclouds,
        merge_decisions=merge_decisions,
    )


@builder.register("SegContactOp")
@taskable_operation_cls
@attrs.frozen
class SegContactOp:
    """Operation to find and write segment contacts with pointclouds and merge decisions."""

    sphere_radius_nm: float
    crop_pad: Sequence[int] = (0, 0, 0)
    min_seg_size_vx: int = 2000
    min_overlap_vx: int = 1000
    min_contact_vx: int = 5
    max_contact_vx: int = 2048
    n_pointcloud_points: int = 2048
    merge_authority: str = "reference_overlap"

    def get_input_resolution(self, dst_resolution: Vec3D[float]) -> Vec3D[float]:
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> SegContactOp:
        return attrs.evolve(self, crop_pad=Vec3D[int](*self.crop_pad) + crop_pad)

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricSegContactLayer,
        segmentation_layer: VolumetricLayer,
        reference_layer: VolumetricLayer,
        affinity_layer: VolumetricLayer,
    ) -> None:
        idx_padded = idx.padded(Vec3D[int](*self.crop_pad))
        resolution = np.array([idx.resolution[0], idx.resolution[1], idx.resolution[2]])

        # Read all layers
        with semaphore("read"):
            seg, reference, aff = _read_layers_parallel(
                segmentation_layer, reference_layer, affinity_layer, idx_padded
            )

        # Compute overlaps and find segments to exclude
        overlap_seg, overlap_ref, overlap_count = _compute_overlaps(seg, reference)
        small_ids = _find_small_segment_ids(seg, self.min_seg_size_vx)
        merger_ids = _find_merger_segment_ids(
            overlap_seg, overlap_ref, overlap_count, self.min_overlap_vx
        )
        unclaimed_ids = _find_unclaimed_segment_ids(overlap_seg, overlap_count, self.min_overlap_vx)
        exclude_ids = small_ids | merger_ids | unclaimed_ids

        # Build seg_to_ref mapping for merge decisions
        seg_to_ref = _build_seg_to_ref(
            overlap_seg, overlap_ref, overlap_count, self.min_overlap_vx
        )

        # Blackout excluded segments
        seg = _blackout_segments(seg, exclude_ids)

        # Find contacts
        seg_a, seg_b, aff_vals, x, y, z = _find_contacts(seg, aff, idx_padded.start)
        if len(seg_a) == 0:
            return

        # Filter to kernel region
        seg_a, seg_b, aff_vals, x, y, z = _filter_pairs_to_kernel(
            seg_a, seg_b, aff_vals, x, y, z, idx_padded.start, seg.shape, self.crop_pad
        )
        if len(seg_a) == 0:
            return

        # Filter pairs by contact count
        contact_counts = _compute_contact_counts(seg_a, seg_b)
        valid_pairs: list[tuple[int, int]] = []
        segs_needing_mesh: set[int] = set()
        for (a, b), count in contact_counts.items():
            if self.min_contact_vx <= count <= self.max_contact_vx:
                valid_pairs.append((a, b))
                segs_needing_mesh.update([a, b])

        if not valid_pairs:
            return

        # Build contact lookup
        contact_data = _build_contact_lookup(seg_a, seg_b, x, y, z, aff_vals)

        # Download meshes
        mesh_cv = CloudVolume(segmentation_layer.backend.name, use_https=True, progress=False)
        bbox_start = (
            np.array([idx_padded.start[0], idx_padded.start[1], idx_padded.start[2]]) * resolution
        )
        bbox_end = (
            np.array([idx_padded.stop[0], idx_padded.stop[1], idx_padded.stop[2]]) * resolution
        )
        meshes = _download_and_clip_meshes(mesh_cv, list(segs_needing_mesh), bbox_start, bbox_end)

        # Generate SegContact objects
        contacts: list[SegContact] = []
        for contact_id, (seg_a_id, seg_b_id) in enumerate(valid_pairs):
            contact = _generate_seg_contact(
                contact_id=contact_id,
                seg_a_id=seg_a_id,
                seg_b_id=seg_b_id,
                meshes=meshes,
                contact_data=contact_data,
                seg_to_ref=seg_to_ref,
                resolution=resolution,
                sphere_radius_nm=self.sphere_radius_nm,
                n_pointcloud_points=self.n_pointcloud_points,
                merge_authority=self.merge_authority,
            )
            if contact is not None:
                contacts.append(contact)

        if contacts:
            with semaphore("write"):
                dst[idx] = contacts

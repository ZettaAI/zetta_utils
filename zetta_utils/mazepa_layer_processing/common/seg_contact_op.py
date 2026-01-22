from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import attrs
import cc3d
import numpy as np
import pandas as pd
import trimesh
from cloudvolume import CloudVolume
from cloudvolume.exceptions import MeshDecodeError
from scipy.spatial.distance import cdist

from zetta_utils import builder, log
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.layer.volumetric.seg_contact import (
    SegContact,
    VolumetricSegContactLayer,
)
from zetta_utils.mazepa import taskable_operation_cls
from zetta_utils.mazepa.semaphores import semaphore

logger = log.get_logger("zetta_utils")


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
    seg_ids: np.ndarray, counts: np.ndarray, min_overlap_vx: int, all_seg_ids: set[int]
) -> set[int]:
    """Find segments without sufficient reference overlap.

    This includes both:
    1. Segments with some overlap but below min_overlap_vx threshold
    2. Segments with ZERO overlap (not present in seg_ids at all)
    """
    seg_max_overlap: dict[int, int] = defaultdict(int)
    for seg, cnt in zip(seg_ids, counts):
        seg_max_overlap[int(seg)] = max(seg_max_overlap[int(seg)], int(cnt))

    # Segments with insufficient overlap
    insufficient_overlap = {
        seg for seg, max_ovl in seg_max_overlap.items() if max_ovl < min_overlap_vx
    }

    # Segments with ZERO overlap (not in seg_ids at all)
    segs_with_any_overlap = set(int(s) for s in seg_ids)
    zero_overlap = all_seg_ids - segs_with_any_overlap - {0}

    return insufficient_overlap | zero_overlap


def _build_seg_to_ref(
    seg_ids: np.ndarray, ref_ids: np.ndarray, counts: np.ndarray, min_overlap_vx: int
) -> dict[int, set[int]]:
    """Build mapping from segment to reference CCs it overlaps with."""
    result: dict[int, set[int]] = defaultdict(set)
    for seg, ref, cnt in zip(seg_ids, ref_ids, counts):
        if cnt >= min_overlap_vx:
            result[int(seg)].add(int(ref))
    return result


def _compute_seg_to_ref_by_segment(
    seg: np.ndarray, reference: np.ndarray, min_overlap_vx: int
) -> dict[int, set[int]]:
    """Build mapping from segment to reference segment IDs (not CCs) it overlaps with.

    This uses raw reference segment IDs, not connected components, so that
    merge decisions work correctly when a reference segment is non-contiguous
    within the chunk.
    """
    flat_seg = seg.ravel()
    flat_ref = reference.ravel()
    valid_mask = (flat_seg != 0) & (flat_ref != 0)
    df = pd.DataFrame({"seg": flat_seg[valid_mask], "ref": flat_ref[valid_mask]})
    counts_df = df.groupby(["seg", "ref"]).size().reset_index(name="count")

    # Use numpy arrays directly to avoid precision loss from iterrows() float conversion
    seg_arr = counts_df["seg"].values
    ref_arr = counts_df["ref"].values
    count_arr = counts_df["count"].values

    result: dict[int, set[int]] = defaultdict(set)
    for i in range(len(counts_df)):
        if count_arr[i] >= min_overlap_vx:
            result[int(seg_arr[i])].add(int(ref_arr[i]))
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


def _filter_pairs_touching_boundary(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    aff: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    start: Vec3D,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exclude pairs that have any contact touching the padded boundary."""
    padded_start = np.array([start[0], start[1], start[2]])
    padded_end = padded_start + np.array(shape)
    on_boundary = (
        (x <= padded_start[0])
        | (x >= padded_end[0] - 1)
        | (y <= padded_start[1])
        | (y >= padded_end[1] - 1)
        | (z <= padded_start[2])
        | (z >= padded_end[2] - 1)
    )
    pairs_on_boundary: set[tuple[int, int]] = set()
    for a, b, on_b in zip(seg_a, seg_b, on_boundary):
        if on_b:
            pairs_on_boundary.add((int(a), int(b)))
    keep = np.array([(int(a), int(b)) not in pairs_on_boundary for a, b in zip(seg_a, seg_b)])
    return seg_a[keep], seg_b[keep], aff[keep], x[keep], y[keep], z[keep]


def _filter_pairs_by_com(
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
    """Exclude pairs whose affinity-weighted COM falls outside the kernel region."""
    kernel_start = np.array([start[0], start[1], start[2]]) + np.array(crop_pad)
    kernel_end = np.array([start[0], start[1], start[2]]) + np.array(shape) - np.array(crop_pad)

    # Vectorized COM computation using pandas groupby
    df = pd.DataFrame({"a": seg_a, "b": seg_b, "x": x, "y": y, "z": z, "aff": aff})
    df["wx"] = df["x"] * df["aff"]
    df["wy"] = df["y"] * df["aff"]
    df["wz"] = df["z"] * df["aff"]

    grouped = df.groupby(["a", "b"]).agg(
        {
            "wx": "sum",
            "wy": "sum",
            "wz": "sum",
            "aff": "sum",
            "x": "mean",
            "y": "mean",
            "z": "mean",
        }
    )

    # Compute COM: weighted if aff_sum > 0, else unweighted mean
    aff_sum = grouped["aff"].values
    has_aff = aff_sum > 0
    com_x = np.where(has_aff, grouped["wx"].values / aff_sum, grouped["x"].values)
    com_y = np.where(has_aff, grouped["wy"].values / aff_sum, grouped["y"].values)
    com_z = np.where(has_aff, grouped["wz"].values / aff_sum, grouped["z"].values)

    # Check which pairs are inside kernel
    inside = (
        (com_x >= kernel_start[0])
        & (com_x < kernel_end[0])
        & (com_y >= kernel_start[1])
        & (com_y < kernel_end[1])
        & (com_z >= kernel_start[2])
        & (com_z < kernel_end[2])
    )

    # Build set of pairs inside kernel
    pairs_inside = set(grouped.index[inside].tolist())
    keep = np.array([(int(a), int(b)) in pairs_inside for a, b in zip(seg_a, seg_b)])
    return seg_a[keep], seg_b[keep], aff[keep], x[keep], y[keep], z[keep]


def _compute_contact_counts(seg_a: np.ndarray, seg_b: np.ndarray) -> dict[tuple[int, int], int]:
    """Count contacts per segment pair."""
    df = pd.DataFrame({"a": seg_a, "b": seg_b})
    counts_df = df.groupby(["a", "b"]).size()
    return {(int(a), int(b)): int(cnt) for (a, b), cnt in counts_df.items()}


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


def _filter_by_mean_affinity(
    contact_data: dict[tuple[int, int], list[tuple[float, float, float, float]]],
    min_affinity: float,
) -> dict[tuple[int, int], list[tuple[float, float, float, float]]]:
    """Filter out contact pairs whose mean affinity is below threshold."""
    if min_affinity <= 0.0:
        return contact_data
    result = {}
    for pair, contacts in contact_data.items():
        mean_aff = sum(c[3] for c in contacts) / len(contacts)
        if mean_aff >= min_affinity:
            result[pair] = contacts
    return result


def _download_meshes(cv: CloudVolume, segment_ids: list[int]) -> dict[int, trimesh.Trimesh]:
    """Download meshes without clipping (clipping done per-contact to sphere)."""
    if not segment_ids:
        return {}

    result: dict[int, trimesh.Trimesh] = {}
    failed_ids: list[int] = []

    for seg_id in segment_ids:
        try:
            meshes = cv.mesh.get([seg_id], progress=False)
            mesh_obj = meshes.get(seg_id)
            if mesh_obj is None or len(mesh_obj.vertices) == 0 or len(mesh_obj.faces) == 0:
                continue
            result[seg_id] = trimesh.Trimesh(vertices=mesh_obj.vertices, faces=mesh_obj.faces)
        except MeshDecodeError:
            failed_ids.append(seg_id)

    if failed_ids:
        print(
            f"Mesh download: {len(failed_ids)}/{len(segment_ids)} "
            f"({100*len(failed_ids)/len(segment_ids):.1f}%) failed with MeshDecodeError"
        )

    return result


def _select_components_near_contacts(
    components: list[trimesh.Trimesh],
    contact_points: np.ndarray,
    touch_threshold: float = 100.0,
) -> trimesh.Trimesh:
    """Select and merge components that are within touch_threshold of contact points."""
    touching = [
        c
        for c in components
        if len(c.vertices) > 0 and cdist(c.vertices, contact_points).min() <= touch_threshold
    ]
    if not touching:
        return max(components, key=lambda c: len(c.faces))
    if len(touching) == 1:
        return touching[0]
    return trimesh.util.concatenate(touching)


def _select_best_component(
    components: list[trimesh.Trimesh],
    contact_points: np.ndarray | None,
) -> trimesh.Trimesh | None:
    """Select best component(s) from a list based on contact points."""
    if len(components) == 0:
        return None
    if len(components) == 1:
        return components[0]
    if contact_points is None or len(contact_points) == 0:
        return max(components, key=lambda c: len(c.faces))
    return _select_components_near_contacts(components, contact_points)


def _crop_mesh_to_sphere(
    mesh: trimesh.Trimesh,
    center: np.ndarray,
    radius: float,
    contact_points: np.ndarray | None = None,
) -> trimesh.Trimesh | None:
    """Clip mesh to sphere, keeping only components within touch_threshold of contact points."""
    vertex_dists = np.linalg.norm(mesh.vertices - center, axis=1)
    vertex_inside = vertex_dists <= radius
    face_inside = vertex_inside[mesh.faces].all(axis=1)

    if not face_inside.any():
        return None

    submesh_result = mesh.submesh([face_inside], append=True)
    cropped = submesh_result[0] if isinstance(submesh_result, list) else submesh_result

    if cropped is None or len(cropped.faces) == 0:
        return None

    components = cropped.split(only_watertight=False)
    return _select_best_component(components, contact_points)


def _sample_mesh_points(mesh: trimesh.Trimesh | None, n: int) -> np.ndarray:
    """Sample N points from mesh surface, area-weighted."""
    if mesh is None or len(mesh.faces) == 0:
        return np.zeros((n, 3), dtype=np.float32)
    # Use deterministic seed based on mesh geometry
    seed = int(np.abs(mesh.vertices.sum() * 1000)) % (2**31)
    rng = np.random.RandomState(seed)
    result = trimesh.sample.sample_surface(mesh, n, seed=rng)
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


def _make_contact_faces_array(
    contacts: list[tuple[float, float, float, float]], resolution: np.ndarray
) -> np.ndarray:
    """Create contact faces array (N, 4) with x, y, z, affinity in nm."""
    x = np.array([c[0] for c in contacts]) * resolution[0]
    y = np.array([c[1] for c in contacts]) * resolution[1]
    z = np.array([c[2] for c in contacts]) * resolution[2]
    aff = np.array([c[3] for c in contacts])
    return np.stack([x, y, z, aff], axis=1).astype(np.float32)


def _build_voxel_spatial_hash(
    voxels: np.ndarray,
) -> dict[tuple[int, int, int], list[int]]:
    """Build spatial hash mapping voxel coordinates to point indices."""
    coord_to_indices: dict[tuple[int, int, int], list[int]] = {}
    for i in range(len(voxels)):
        key = (int(voxels[i, 0]), int(voxels[i, 1]), int(voxels[i, 2]))
        if key not in coord_to_indices:
            coord_to_indices[key] = []
        coord_to_indices[key].append(i)
    return coord_to_indices


def _get_unvisited_neighbors(
    voxel_coord: tuple[int, int, int],
    coord_to_indices: dict[tuple[int, int, int], list[int]],
    visited: np.ndarray,
) -> list[int]:
    """Get unvisited neighbor indices for a voxel using 6-connectivity."""
    offsets = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    neighbors = []
    cx, cy, cz = voxel_coord
    for dx, dy, dz in offsets:
        neighbor_key = (cx + dx, cy + dy, cz + dz)
        for idx in coord_to_indices.get(neighbor_key, []):
            if not visited[idx]:
                neighbors.append(idx)
    return neighbors


def _compute_contact_connected_components(
    contact_faces: np.ndarray,
    resolution: np.ndarray,
) -> list[np.ndarray]:
    """Compute connected components of contact faces using 6-connectivity."""
    if len(contact_faces) == 0:
        return []

    voxels = np.round(contact_faces[:, :3] / resolution).astype(np.int64)
    coord_to_indices = _build_voxel_spatial_hash(voxels)

    visited = np.zeros(len(voxels), dtype=bool)
    components = []

    for start in range(len(voxels)):
        if visited[start]:
            continue

        component = [start]
        visited[start] = True
        queue = [start]

        while queue:
            current = queue.pop(0)
            voxel_coord = (
                int(voxels[current, 0]),
                int(voxels[current, 1]),
                int(voxels[current, 2]),
            )
            for neighbor_idx in _get_unvisited_neighbors(voxel_coord, coord_to_indices, visited):
                visited[neighbor_idx] = True
                component.append(neighbor_idx)
                queue.append(neighbor_idx)

        components.append(np.array(component))

    return components


def _find_contact_center(contact_faces: np.ndarray, resolution: np.ndarray) -> np.ndarray:
    """Find contact center as the point closest to mean of largest component."""
    xyz = contact_faces[:, :3]
    components = _compute_contact_connected_components(contact_faces, resolution)
    if components:
        xyz = xyz[max(components, key=len)]
    mean_pos = xyz.mean(axis=0)
    return xyz[np.argmin(np.linalg.norm(xyz - mean_pos, axis=1))]


def _sample_sphere_voxels(
    center_nm: np.ndarray,
    radius: float,
    seg_volume: np.ndarray,
    seg_start_nm: np.ndarray,
    resolution: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample voxels within sphere, return local indices and nm coordinates."""
    center_vx = np.round(center_nm / resolution).astype(np.int32)
    seg_start_vx = (seg_start_nm / resolution).astype(np.int32)
    pad_vx = (radius / resolution + 1).astype(np.int32)

    ranges = [
        np.arange(
            max(0, center_vx[i] - pad_vx[i] - seg_start_vx[i]),
            min(seg_volume.shape[i], center_vx[i] + pad_vx[i] + 1 - seg_start_vx[i]),
            dtype=np.int32,
        )
        for i in range(3)
    ]
    if any(len(r) == 0 for r in ranges):
        return np.array([]), np.array([]), np.array([]), np.array([])

    grids = np.meshgrid(*ranges, indexing="ij")
    local_vx = np.column_stack([g.ravel() for g in grids])
    global_vx = local_vx + seg_start_vx
    voxel_nm = global_vx.astype(np.float32) * resolution

    dist_sq = np.sum((voxel_nm - center_nm) ** 2, axis=1)
    in_sphere = dist_sq <= radius * radius

    return (
        local_vx[in_sphere],
        voxel_nm[in_sphere],
        seg_volume[local_vx[in_sphere, 0], local_vx[in_sphere, 1], local_vx[in_sphere, 2]],
        in_sphere,
    )


def _voxel_closest_to_mean(voxel_nm: np.ndarray, mask: np.ndarray) -> Vec3D[float] | None:
    """Find voxel closest to mean of masked voxels, return as Vec3D."""
    if not mask.any():
        return None
    pts = voxel_nm[mask]
    mean = pts.mean(axis=0)
    idx = np.argmin(np.sum((pts - mean) ** 2, axis=1))
    return Vec3D(float(pts[idx, 0]), float(pts[idx, 1]), float(pts[idx, 2]))


def _compute_representative_points(
    contact_faces: np.ndarray,
    seg_a_id: int,
    seg_b_id: int,
    seg_volume: np.ndarray,
    seg_start_nm: np.ndarray,
    resolution: np.ndarray,
) -> dict[int, Vec3D[float]]:
    """Compute representative points as voxels closest to segment centers within sphere."""
    contact_center = _find_contact_center(contact_faces, resolution)
    fallback = Vec3D(float(contact_center[0]), float(contact_center[1]), float(contact_center[2]))

    local_vx, voxel_nm, seg_ids, _ = _sample_sphere_voxels(
        contact_center, 200.0, seg_volume, seg_start_nm, resolution
    )
    if len(local_vx) == 0:
        return {seg_a_id: fallback, seg_b_id: fallback}

    pt_a = _voxel_closest_to_mean(voxel_nm, seg_ids == seg_a_id)
    pt_b = _voxel_closest_to_mean(voxel_nm, seg_ids == seg_b_id)

    return {seg_a_id: pt_a or fallback, seg_b_id: pt_b or fallback}


def _generate_seg_contact(
    contact_id: int,
    seg_a_id: int,
    seg_b_id: int,
    meshes: dict[int, trimesh.Trimesh],
    contact_data: dict[tuple[int, int], list[tuple[float, float, float, float]]],
    seg_to_ref: dict[int, set[int]],
    resolution: np.ndarray,
    pointcloud_configs: list[tuple[float, int]],
    merge_authority: str,
    seg_volume: np.ndarray,
    seg_start_nm: np.ndarray,
) -> SegContact | None:
    """Generate a single SegContact for a contact pair."""
    mesh_a, mesh_b = meshes.get(seg_a_id), meshes.get(seg_b_id)
    if mesh_a is None or mesh_b is None:
        return None

    com = _compute_affinity_weighted_com(contact_data[(seg_a_id, seg_b_id)], resolution)
    contact_faces = _make_contact_faces_array(contact_data[(seg_a_id, seg_b_id)], resolution)

    local_pointclouds = _generate_pointclouds(
        mesh_a, mesh_b, seg_a_id, seg_b_id, com, contact_faces[:, :3], pointcloud_configs
    )
    if not local_pointclouds:
        return None

    return SegContact(
        id=contact_id,
        seg_a=seg_a_id,
        seg_b=seg_b_id,
        com=Vec3D(float(com[0]), float(com[1]), float(com[2])),
        contact_faces=contact_faces,
        local_pointclouds=local_pointclouds,
        merge_decisions={
            merge_authority: bool(
                seg_to_ref.get(seg_a_id, set()) & seg_to_ref.get(seg_b_id, set())
            )
        },
        representative_points=_compute_representative_points(
            contact_faces, seg_a_id, seg_b_id, seg_volume, seg_start_nm, resolution
        ),
    )


def _generate_pointclouds(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    seg_a_id: int,
    seg_b_id: int,
    com: np.ndarray,
    contact_points_xyz: np.ndarray,
    pointcloud_configs: list[tuple[float, int]],
) -> dict[tuple[int, int], dict[int, np.ndarray]]:
    """Generate pointclouds for all configs."""
    result: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    for radius_nm, n_points in pointcloud_configs:
        mesh_a_cropped = _crop_mesh_to_sphere(mesh_a, com, radius_nm, contact_points_xyz)
        mesh_b_cropped = _crop_mesh_to_sphere(mesh_b, com, radius_nm, contact_points_xyz)
        if mesh_a_cropped is None or mesh_b_cropped is None:
            continue
        result[(int(radius_nm), n_points)] = {
            seg_a_id: _sample_mesh_points(mesh_a_cropped, n_points),
            seg_b_id: _sample_mesh_points(mesh_b_cropped, n_points),
        }
    return result


@builder.register("SegContactOp")
@taskable_operation_cls
@attrs.frozen
class SegContactOp:
    """Operation to find and write segment contacts with pointclouds and merge decisions."""

    crop_pad: Sequence[int] = (0, 0, 0)
    min_seg_size_vx: int = 2000
    min_overlap_vx: int = 1000
    min_contact_vx: int = 5
    max_contact_vx: int = 2048
    min_affinity: float = 0.0
    merge_authority: str = "reference_overlap"
    ids_per_chunk: int = 10000

    def get_input_resolution(self, dst_resolution: Vec3D[float]) -> Vec3D[float]:
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> SegContactOp:
        return attrs.evolve(self, crop_pad=Vec3D[int](*self.crop_pad) + crop_pad)

    def __call__(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self,
        idx: VolumetricIndex,
        dst: VolumetricSegContactLayer,
        segmentation_layer: VolumetricLayer,
        reference_layer: VolumetricLayer,
        affinity_layer: VolumetricLayer,
    ) -> None:
        t_start = time.time()
        coord_str = f"{int(idx.start[0])}_{int(idx.start[1])}_{int(idx.start[2])}"
        idx_padded = idx.padded(Vec3D[int](*self.crop_pad))
        resolution = np.array([idx.resolution[0], idx.resolution[1], idx.resolution[2]])

        # Read pointcloud configs from destination layer info file
        pointcloud_configs = dst.backend.get_pointcloud_configs()

        # Read all layers
        t0 = time.time()
        with semaphore("read"):
            seg, reference, aff = _read_layers_parallel(
                segmentation_layer, reference_layer, affinity_layer, idx_padded
            )
        print(f"[{coord_str}] Read layers: {time.time() - t0:.1f}s", flush=True)

        # Compute overlaps and find segments to exclude
        t0 = time.time()
        overlap_seg, overlap_ref, overlap_count = _compute_overlaps(seg, reference)
        all_seg_ids = set(int(s) for s in np.unique(seg) if s != 0)
        small_ids = _find_small_segment_ids(seg, self.min_seg_size_vx)
        merger_ids = _find_merger_segment_ids(
            overlap_seg, overlap_ref, overlap_count, self.min_overlap_vx
        )
        unclaimed_ids = _find_unclaimed_segment_ids(
            overlap_seg, overlap_count, self.min_overlap_vx, all_seg_ids
        )
        exclude_ids = small_ids | merger_ids | unclaimed_ids
        print(
            f"[{coord_str}] Overlaps/filtering: {time.time() - t0:.1f}s "
            f"(excl {len(small_ids)} small, {len(merger_ids)} merger, "
            f"{len(unclaimed_ids)} unclaimed)",
            flush=True,
        )

        # Build seg_to_ref mapping for merge decisions using raw reference segment IDs
        # (not connected components, so non-contiguous reference segments are handled correctly)
        seg_to_ref = _compute_seg_to_ref_by_segment(seg, reference, self.min_overlap_vx)

        # Blackout excluded segments
        seg = _blackout_segments(seg, exclude_ids)

        # Find contacts
        t0 = time.time()
        seg_a, seg_b, aff_vals, x, y, z = _find_contacts(seg, aff, idx_padded.start)
        if len(seg_a) == 0:
            print(f"[{coord_str}] No contacts found, skipping", flush=True)
            return
        print(
            f"[{coord_str}] Find contacts: {time.time() - t0:.1f}s ({len(seg_a)} voxels)",
            flush=True,
        )

        # Filter out pairs touching padded boundary (may have incomplete contacts)
        seg_a, seg_b, aff_vals, x, y, z = _filter_pairs_touching_boundary(
            seg_a, seg_b, aff_vals, x, y, z, idx_padded.start, seg.shape
        )
        if len(seg_a) == 0:
            print(f"[{coord_str}] All pairs on boundary, skipping", flush=True)
            return

        # Filter out pairs with COM outside kernel region
        seg_a, seg_b, aff_vals, x, y, z = _filter_pairs_by_com(
            seg_a, seg_b, aff_vals, x, y, z, idx_padded.start, seg.shape, self.crop_pad
        )
        if len(seg_a) == 0:
            print(f"[{coord_str}] All pairs outside kernel, skipping", flush=True)
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
            print(f"[{coord_str}] No valid pairs after filtering, skipping", flush=True)
            return
        print(
            f"[{coord_str}] Valid pairs: {len(valid_pairs)}, "
            f"segs needing mesh: {len(segs_needing_mesh)}",
            flush=True,
        )

        # Build contact lookup
        contact_data = _build_contact_lookup(seg_a, seg_b, x, y, z, aff_vals)

        # Filter by mean affinity
        if self.min_affinity > 0.0:
            n_before = len(contact_data)
            contact_data = _filter_by_mean_affinity(contact_data, self.min_affinity)
            n_after = len(contact_data)
            if n_after < n_before:
                print(
                    f"[{coord_str}] Filtered by min_affinity={self.min_affinity}: "
                    f"{n_before} -> {n_after} pairs",
                    flush=True,
                )
            if not contact_data:
                print(f"[{coord_str}] No pairs after affinity filter, skipping", flush=True)
                return
            # Update valid_pairs to only include pairs that passed affinity filter
            valid_pairs = [(a, b) for (a, b) in valid_pairs if (a, b) in contact_data]
            segs_needing_mesh = set()
            for a, b in valid_pairs:
                segs_needing_mesh.update([a, b])

        # Download meshes (no bbox clipping - sphere cropping done per-contact)
        t0 = time.time()
        mesh_cv = CloudVolume(segmentation_layer.backend.name, use_https=True, progress=False)
        meshes = _download_meshes(mesh_cv, list(segs_needing_mesh))
        print(
            f"[{coord_str}] Download meshes: {time.time() - t0:.1f}s ({len(meshes)} meshes)",
            flush=True,
        )

        # Generate SegContact objects
        t0 = time.time()
        id_offset = idx.chunk_id * self.ids_per_chunk
        seg_start_nm = np.array(
            [
                idx_padded.start[0] * resolution[0],
                idx_padded.start[1] * resolution[1],
                idx_padded.start[2] * resolution[2],
            ]
        )
        contacts: list[SegContact] = []
        for local_id, (seg_a_id, seg_b_id) in enumerate(valid_pairs):
            contact = _generate_seg_contact(
                contact_id=id_offset + local_id,
                seg_a_id=seg_a_id,
                seg_b_id=seg_b_id,
                meshes=meshes,
                contact_data=contact_data,
                seg_to_ref=seg_to_ref,
                resolution=resolution,
                pointcloud_configs=pointcloud_configs,
                merge_authority=self.merge_authority,
                seg_volume=seg,
                seg_start_nm=seg_start_nm,
            )
            if contact is not None:
                contacts.append(contact)
        print(
            f"[{coord_str}] Generate contacts: {time.time() - t0:.1f}s ({len(contacts)} contacts)",
            flush=True,
        )

        if contacts:
            # Count merge decisions
            n_merge = sum(
                1
                for c in contacts
                if c.merge_decisions and c.merge_decisions.get(self.merge_authority, False)
            )
            n_no_merge = len(contacts) - n_merge
            print(
                f"[{coord_str}] Merge decisions: {n_merge} MERGE, {n_no_merge} NO-MERGE "
                f"(total {len(contacts)})",
                flush=True,
            )

            t0 = time.time()
            with semaphore("write"):
                dst[idx] = contacts
            print(f"[{coord_str}] Write: {time.time() - t0:.1f}s", flush=True)

        print(f"[{coord_str}] Total: {time.time() - t_start:.1f}s", flush=True)


@builder.register("AddPointcloudsOp")
@taskable_operation_cls
@attrs.frozen
class AddPointcloudsOp:
    """Add new pointcloud configs to existing contacts without recomputing contacts."""

    pointcloud_configs: Sequence[tuple[float, int]]  # [(radius_nm, n_points), ...]

    def get_input_resolution(self, dst_resolution: Vec3D[float]) -> Vec3D[float]:
        return dst_resolution

    def __call__(  # pylint: disable=too-many-locals
        self,
        idx: VolumetricIndex,
        dst: VolumetricSegContactLayer,
        segmentation_layer: VolumetricLayer,
    ) -> None:
        t_start = time.time()
        coord_str = f"{int(idx.start[0])}_{int(idx.start[1])}_{int(idx.start[2])}"

        # Convert idx to chunk_idx
        chunk_idx = dst.backend.com_to_chunk_idx(idx.bbox.start)

        # Read existing contacts (just contact data, no pointclouds needed)
        t0 = time.time()
        with semaphore("read"):
            contacts_data = dst.backend._read_contacts_chunk(chunk_idx)
        if not contacts_data:
            print(f"[{coord_str}] No contacts in chunk, skipping", flush=True)
            return
        print(
            f"[{coord_str}] Read contacts: {time.time() - t0:.1f}s "
            f"({len(contacts_data)} contacts)",
            flush=True,
        )

        # Get unique segment IDs
        segs_needing_mesh: set[int] = set()
        for c in contacts_data:
            segs_needing_mesh.add(c["seg_a"])
            segs_needing_mesh.add(c["seg_b"])

        # Download meshes
        t0 = time.time()
        mesh_cv = CloudVolume(segmentation_layer.backend.name, use_https=True, progress=False)
        meshes = _download_meshes(mesh_cv, list(segs_needing_mesh))
        print(
            f"[{coord_str}] Download meshes: {time.time() - t0:.1f}s ({len(meshes)} meshes)",
            flush=True,
        )

        # Generate pointclouds for each config
        pointclouds_by_config: dict[
            tuple[int, int], list[tuple[int, int, int, np.ndarray, np.ndarray]]
        ] = {}

        t0 = time.time()
        for contact in contacts_data:
            contact_id = contact["id"]
            seg_a = contact["seg_a"]
            seg_b = contact["seg_b"]
            com = contact["com"]
            contact_faces = contact["contact_faces"]

            mesh_a = meshes.get(seg_a)
            mesh_b = meshes.get(seg_b)
            if mesh_a is None or mesh_b is None:
                continue

            com_np = np.array([com[0], com[1], com[2]])
            contact_points_xyz = contact_faces[:, :3] if contact_faces.shape[0] > 0 else None

            for radius_nm, n_points in self.pointcloud_configs:
                config_tuple = (int(radius_nm), n_points)

                mesh_a_cropped = _crop_mesh_to_sphere(
                    mesh_a, com_np, radius_nm, contact_points_xyz
                )
                mesh_b_cropped = _crop_mesh_to_sphere(
                    mesh_b, com_np, radius_nm, contact_points_xyz
                )
                if mesh_a_cropped is None or mesh_b_cropped is None:
                    continue

                pointcloud_a = _sample_mesh_points(mesh_a_cropped, n_points)
                pointcloud_b = _sample_mesh_points(mesh_b_cropped, n_points)

                if config_tuple not in pointclouds_by_config:
                    pointclouds_by_config[config_tuple] = []
                pointclouds_by_config[config_tuple].append(
                    (contact_id, seg_a, seg_b, pointcloud_a, pointcloud_b)
                )

        print(f"[{coord_str}] Generate pointclouds: {time.time() - t0:.1f}s", flush=True)

        # Write pointcloud chunks
        t0 = time.time()
        with semaphore("write"):
            for config_tuple, entries in pointclouds_by_config.items():
                dst.backend._write_pointcloud_chunk(chunk_idx, config_tuple, entries)
        print(
            f"[{coord_str}] Write pointclouds: {time.time() - t0:.1f}s "
            f"({len(pointclouds_by_config)} configs)",
            flush=True,
        )

        print(f"[{coord_str}] Total: {time.time() - t_start:.1f}s", flush=True)


@builder.register("ContactMergeOp")
@taskable_operation_cls
@attrs.frozen
class ContactMergeOp:
    """Operation to run contact merge inference and write merge_probabilities.

    Reads contacts from src layer, runs PointNet model inference,
    and writes updated contacts with merge_probabilities to dst layer.

    The conversion parameters must match the validation dataset configuration
    used during training to ensure consistent data representation.
    """

    model_path: str
    authority_name: str
    apply_sigmoid: bool = True
    include_contact_faces: bool = False
    contact_label: float | None = 0.0
    affinity_channel_mode: str | None = None
    config_key: tuple[int, int] | None = None

    def get_input_resolution(self, dst_resolution: Vec3D[float]) -> Vec3D[float]:
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> "ContactMergeOp":
        return self  # No crop pad needed for this op

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricSegContactLayer,
        src: VolumetricSegContactLayer,
    ) -> None:
        import torch

        from zetta_utils import convnet
        from zetta_utils.layer.volumetric.seg_contact.tensor_utils import contacts_to_tensor

        t_start = time.time()
        coord_str = f"{int(idx.start[0])}_{int(idx.start[1])}_{int(idx.start[2])}"

        # Read contacts from source layer
        t0 = time.time()
        with semaphore("read"):
            contacts = src[idx]
        if not contacts:
            logger.info(f"[{coord_str}] No contacts in chunk, skipping")
            return
        logger.info(
            f"[{coord_str}] Read contacts: {time.time() - t0:.1f}s ({len(contacts)} contacts)"
        )

        # Convert to tensor for model input
        t0 = time.time()
        tensor, valid_indices = contacts_to_tensor(
            contacts,
            config_key=self.config_key,
            include_contact_faces=self.include_contact_faces,
            contact_label=self.contact_label,
            affinity_channel_mode=self.affinity_channel_mode,
        )
        logger.info(
            f"[{coord_str}] Convert to tensor: {time.time() - t0:.1f}s "
            f"({len(valid_indices)} valid contacts)"
        )

        if tensor.shape[0] == 0:
            logger.info(f"[{coord_str}] No valid contacts with pointclouds, skipping")
            return

        # Run model inference
        t0 = time.time()
        with torch.no_grad():
            output = convnet.utils.load_and_run_model(path=self.model_path, data_in=tensor)

        if self.apply_sigmoid:
            output = torch.sigmoid(output)

        probs = output.squeeze().cpu()
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
        logger.info(f"[{coord_str}] Model inference: {time.time() - t0:.1f}s")

        # Update contacts with merge_probabilities
        t0 = time.time()
        result = list(contacts)
        for i, contact_idx in enumerate(valid_indices):
            contact = result[contact_idx]
            prob = float(probs[i].item())

            if contact.merge_probabilities is None:
                new_probs = {self.authority_name: prob}
            else:
                new_probs = dict(contact.merge_probabilities)
                new_probs[self.authority_name] = prob

            result[contact_idx] = SegContact(
                id=contact.id,
                seg_a=contact.seg_a,
                seg_b=contact.seg_b,
                com=contact.com,
                contact_faces=contact.contact_faces,
                representative_points=contact.representative_points,
                local_pointclouds=contact.local_pointclouds,
                merge_decisions=contact.merge_decisions,
                merge_probabilities=new_probs,
                partner_metadata=contact.partner_metadata,
            )
        logger.info(f"[{coord_str}] Update contacts: {time.time() - t0:.1f}s")

        # Write updated contacts
        t0 = time.time()
        with semaphore("write"):
            dst[idx] = result
        logger.info(f"[{coord_str}] Write contacts: {time.time() - t0:.1f}s")

        logger.info(f"[{coord_str}] Total ContactMergeOp: {time.time() - t_start:.1f}s")

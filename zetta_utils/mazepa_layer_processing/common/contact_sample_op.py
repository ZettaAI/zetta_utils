from __future__ import annotations

import contextlib
import io
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence

import attrs
import cc3d
import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import trimesh
from cloudvolume import CloudVolume

from zetta_utils import builder
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.mazepa import taskable_operation_cls
from zetta_utils.mazepa.semaphores import semaphore


@dataclass
class FilterParams:
    """Parameters for segment filtering."""

    min_seg_size_vx: int
    min_overlap_vx: int
    min_contact_vx: int
    max_contact_vx: int


@dataclass
class SampleConfig:
    """Configuration for sample generation."""

    resolution: np.ndarray
    sphere_radius_nm: float
    n_pointcloud_points: int
    max_contact_vx: int
    metadata: dict[str, Any]
    coord_str: str


def _read_layers_parallel(
    candidate: VolumetricLayer,
    proofread: VolumetricLayer,
    affinity: VolumetricLayer,
    idx: VolumetricIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def read_layer(layer: VolumetricLayer, index: VolumetricIndex) -> np.ndarray:
        return np.asarray(layer[index])

    with ThreadPoolExecutor(max_workers=3) as executor:
        cand_future = executor.submit(read_layer, candidate, idx)
        proof_future = executor.submit(read_layer, proofread, idx)
        aff_future = executor.submit(read_layer, affinity, idx)
        return cand_future.result().squeeze(), proof_future.result().squeeze(), aff_future.result()


def _compute_overlaps(
    candidate_seg: np.ndarray, proofread_seg: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute overlaps between candidate segments and proofread connected components."""
    cc_proofread = cc3d.connected_components(proofread_seg, connectivity=6)
    flat_candidate = candidate_seg.ravel()
    flat_cc_proof = cc_proofread.ravel()
    valid_mask = (flat_candidate != 0) & (flat_cc_proof != 0)
    df = pd.DataFrame({"cand": flat_candidate[valid_mask], "cc_proof": flat_cc_proof[valid_mask]})
    counts_df = df.groupby(["cand", "cc_proof"]).size().reset_index(name="count")
    return (
        counts_df["cand"].values.astype(np.int64),
        counts_df["cc_proof"].values.astype(np.int64),
        counts_df["count"].values.astype(np.int32),
    )


def _find_small_segment_ids(candidate_seg: np.ndarray, min_seg_size_vx: int) -> set[int]:
    """Find segment IDs with total voxel count below threshold."""
    unique, counts = np.unique(candidate_seg, return_counts=True)
    return {int(seg) for seg, cnt in zip(unique, counts) if seg != 0 and cnt < min_seg_size_vx}


def _find_merger_segment_ids(
    cand_ids: np.ndarray, proof_ids: np.ndarray, counts: np.ndarray, min_overlap_vx: int
) -> set[int]:
    """Find merger segments (overlap 2+ proofread CCs with >= min_overlap each)."""
    cand_to_proof: dict[int, set[int]] = defaultdict(set)
    for cand, proof, cnt in zip(cand_ids, proof_ids, counts):
        if cnt >= min_overlap_vx:
            cand_to_proof[int(cand)].add(int(proof))
    return {cand for cand, proofs in cand_to_proof.items() if len(proofs) >= 2}


def _find_unclaimed_segment_ids(
    cand_ids: np.ndarray, counts: np.ndarray, min_overlap_vx: int
) -> set[int]:
    """Find segments without sufficient proofread overlap."""
    seg_max_overlap: dict[int, int] = defaultdict(int)
    for cand, cnt in zip(cand_ids, counts):
        seg_max_overlap[int(cand)] = max(seg_max_overlap[int(cand)], int(cnt))
    return {seg for seg, max_ovl in seg_max_overlap.items() if max_ovl < min_overlap_vx}


def _build_cand_to_proof(
    cand_ids: np.ndarray, proof_ids: np.ndarray, counts: np.ndarray, min_overlap_vx: int
) -> dict[int, set[int]]:
    """Build mapping from candidate segment to proofread CCs it overlaps with."""
    result: dict[int, set[int]] = defaultdict(set)
    for cand, proof, cnt in zip(cand_ids, proof_ids, counts):
        if cnt >= min_overlap_vx:
            result[int(cand)].add(int(proof))
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
    """Find contacts along one axis. Returns face centers (with 0.5 offset on contact axis)."""
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


def _dedupe_and_average_contacts(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    aff: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average affinities for voxels touching on multiple axes and normalize order."""
    # Normalize segment order BEFORE grouping to avoid duplicates
    swap = seg_a > seg_b
    seg_a_norm = np.where(swap, seg_b, seg_a)
    seg_b_norm = np.where(swap, seg_a, seg_b)
    df = pd.DataFrame(
        {"seg_a": seg_a_norm, "seg_b": seg_b_norm, "x": x, "y": y, "z": z, "aff": aff}
    )
    deduped = df.groupby(["seg_a", "seg_b", "x", "y", "z"], as_index=False).agg({"aff": "mean"})
    return (
        deduped["seg_a"].values,
        deduped["seg_b"].values,
        deduped["aff"].values.astype(np.float32),
        deduped["x"].values,
        deduped["y"].values,
        deduped["z"].values,
    )


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
    # Find pairs with any contact outside kernel
    pairs_outside: set[tuple[int, int]] = set()
    for a, b, out in zip(seg_a, seg_b, outside_kernel):
        if out:
            pairs_outside.add((int(a), int(b)))
    # Keep only contacts from pairs fully inside kernel
    keep = np.array([(int(a), int(b)) not in pairs_outside for a, b in zip(seg_a, seg_b)])
    return seg_a[keep], seg_b[keep], aff[keep], x[keep], y[keep], z[keep]


def _find_contacts(
    candidate_seg: np.ndarray, affinity_raw: np.ndarray, start: Vec3D, crop_pad: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find contacts between segments, using relevant axis affinity for each."""
    sx, sy, sz = int(start[0]), int(start[1]), int(start[2])
    results = []

    t0 = time.time()
    # For each axis, offset by 0.5 on the contact axis to get face center
    # Extra 0.5 on Z axis
    for seg_lo, seg_hi, aff_slice, offset in [
        (
            candidate_seg[:-1],
            candidate_seg[1:],
            affinity_raw[0, 1:],
            (sx + 0.5, sy, sz + 0.5),
        ),  # X
        (
            candidate_seg[:, :-1],
            candidate_seg[:, 1:],
            affinity_raw[1, :, 1:],
            (sx, sy + 0.5, sz + 0.5),
        ),  # Y
        (
            candidate_seg[:, :, :-1],
            candidate_seg[:, :, 1:],
            affinity_raw[2, :, :, 1:],
            (sx, sy, sz + 1.0),
        ),  # Z
    ]:
        r = _find_axis_contacts(seg_lo, seg_hi, aff_slice, offset)
        if len(r[0]) > 0:
            results.append(r)
    print(f"      axis contacts: {time.time() - t0:.1f}s", flush=True)

    if not results:
        empty_i, empty_f = np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        return empty_i, empty_i, empty_f, empty_i, empty_i, empty_i

    t0 = time.time()
    seg_a = np.concatenate([r[0] for r in results])
    seg_b = np.concatenate([r[1] for r in results])
    aff = np.concatenate([r[2] for r in results])
    x, y, z = (
        np.concatenate([r[3] for r in results]),
        np.concatenate([r[4] for r in results]),
        np.concatenate([r[5] for r in results]),
    )
    # Normalize segment order so seg_a < seg_b
    swap = seg_a > seg_b
    seg_a, seg_b = np.where(swap, seg_b, seg_a), np.where(swap, seg_a, seg_b)
    print(f"      concatenate ({len(seg_a)} contacts): {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    result = _filter_pairs_to_kernel(
        seg_a, seg_b, aff, x, y, z, start, candidate_seg.shape, crop_pad
    )
    print(f"      filter kernel ({len(result[0])} after): {time.time() - t0:.1f}s", flush=True)
    return result


def _compute_contact_counts(seg_a: np.ndarray, seg_b: np.ndarray) -> dict[tuple[int, int], int]:
    """Count contacts per segment pair."""
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for a, b in zip(seg_a, seg_b):
        counts[(int(a), int(b))] += 1
    return counts


def _download_and_clip_meshes(
    cv: CloudVolume, segment_ids: list[int], bbox_start: np.ndarray, bbox_end: np.ndarray
) -> dict[int, trimesh.Trimesh]:
    """Download meshes and clip to bounding box, return trimesh objects."""
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


def _sample_mesh_points(mesh: trimesh.Trimesh, n: int) -> np.ndarray:
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


def _make_contact_array(
    contacts: list[tuple[float, float, float, float]], resolution: np.ndarray, max_size: int
) -> tuple[np.ndarray, int]:
    """Create padded contact array from contact list."""
    x = np.array([c[0] for c in contacts]) * resolution[0]
    y = np.array([c[1] for c in contacts]) * resolution[1]
    z = np.array([c[2] for c in contacts]) * resolution[2]
    aff = np.array([c[3] for c in contacts])
    arr = np.stack([x, y, z, aff], axis=1).astype(np.float32)
    n = len(arr)
    padded = np.zeros((max_size, 4), dtype=np.float32)
    padded[:n] = arr
    return padded, n


def _make_empty_table() -> pa.Table:
    """Create empty table with correct schema."""
    return pa.Table.from_pydict(
        {
            "seg_a": pa.array([], type=pa.int64()),
            "seg_b": pa.array([], type=pa.int64()),
            "should_merge": pa.array([], type=pa.int64()),
            "n_contacts": pa.array([], type=pa.int64()),
            "contacts": pa.array([], type=pa.list_(pa.list_(pa.float64()))),
            "pointcloud_a": pa.array([], type=pa.list_(pa.list_(pa.float64()))),
            "pointcloud_b": pa.array([], type=pa.list_(pa.list_(pa.float64()))),
            "chunk_coord": pa.array([], type=pa.list_(pa.int64())),
            "chunk_size": pa.array([], type=pa.list_(pa.int64())),
            "crop_pad": pa.array([], type=pa.list_(pa.int64())),
            "sphere_radius_nm": pa.array([], type=pa.float64()),
            "candidate_path": pa.array([], type=pa.string()),
            "reference_path": pa.array([], type=pa.string()),
            "affinity_path": pa.array([], type=pa.string()),
        }
    )


def _write_feather(table: pa.Table, path: str) -> None:
    """Write feather table to path."""
    buffer = io.BytesIO()
    pf.write_feather(table, buffer, compression="zstd")
    buffer.seek(0)
    with fsspec.open(path, "wb") as f:
        f.write(buffer.read())


def _find_excluded_segments_and_overlaps(
    candidate_seg: np.ndarray,
    proofread_seg: np.ndarray,
    min_seg_size_vx: int,
    min_overlap_vx: int,
    coord_str: str,
) -> tuple[set[int], dict[int, set[int]]]:
    """Compute overlaps, find excluded segments, and build cand_to_proof mapping."""
    overlap_cand, overlap_proof, overlap_count = _compute_overlaps(candidate_seg, proofread_seg)
    small_ids = _find_small_segment_ids(candidate_seg, min_seg_size_vx)
    merger_ids = _find_merger_segment_ids(
        overlap_cand, overlap_proof, overlap_count, min_overlap_vx
    )
    unclaimed_ids = _find_unclaimed_segment_ids(overlap_cand, overlap_count, min_overlap_vx)
    exclude_ids = small_ids | merger_ids | unclaimed_ids
    print(
        f"[{coord_str}] Excluding {len(exclude_ids)} segments: "
        f"{len(small_ids)} small, {len(merger_ids)} mergers, {len(unclaimed_ids)} unclaimed",
        flush=True,
    )
    cand_to_proof = _build_cand_to_proof(
        overlap_cand, overlap_proof, overlap_count, min_overlap_vx
    )
    return exclude_ids, cand_to_proof


def _filter_contact_pairs(
    contact_counts: dict[tuple[int, int], int], min_vx: int, max_vx: int, coord_str: str
) -> tuple[list[tuple[int, int, int]], set[int]]:
    """Filter contact pairs by count and return valid pairs with segments needing meshes."""
    valid_pairs: list[tuple[int, int, int]] = []
    segs_needing_mesh: set[int] = set()
    n_low = n_high = 0
    for (a, b), count in contact_counts.items():
        if count < min_vx:
            n_low += 1
        elif count > max_vx:
            n_high += 1
        else:
            valid_pairs.append((a, b, count))
            segs_needing_mesh.update([a, b])
    print(f"[{coord_str}] Contact pairs: {len(contact_counts)}", flush=True)
    print(f"[{coord_str}]   low contact (<{min_vx}): -{n_low}", flush=True)
    print(f"[{coord_str}]   high contact (>{max_vx}): -{n_high}", flush=True)
    print(f"[{coord_str}]   valid pairs: {len(valid_pairs)}", flush=True)
    return valid_pairs, segs_needing_mesh


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


def _generate_single_sample(
    seg_a_id: int,
    seg_b_id: int,
    meshes: dict[int, trimesh.Trimesh],
    contact_data: dict[tuple[int, int], list[tuple[float, float, float, float]]],
    cand_to_proof: dict[int, set[int]],
    cfg: SampleConfig,
) -> dict[str, Any] | None:
    """Generate a single training sample for a contact pair."""
    mesh_a, mesh_b = meshes.get(seg_a_id), meshes.get(seg_b_id)
    if mesh_a is None or mesh_b is None:
        return None

    contacts = contact_data[(seg_a_id, seg_b_id)]
    com = _compute_affinity_weighted_com(contacts, cfg.resolution)

    mesh_a_cropped = _crop_mesh_to_sphere(mesh_a, com, cfg.sphere_radius_nm)
    mesh_b_cropped = _crop_mesh_to_sphere(mesh_b, com, cfg.sphere_radius_nm)
    if mesh_a_cropped is None or mesh_b_cropped is None:
        return None

    if not _all_contacts_in_sphere(contacts, com, cfg.sphere_radius_nm, cfg.resolution):
        return None

    contacts_padded, n_contacts = _make_contact_array(contacts, cfg.resolution, cfg.max_contact_vx)
    label = 1 if cand_to_proof.get(seg_a_id, set()) & cand_to_proof.get(seg_b_id, set()) else 0

    return {
        "seg_a": seg_a_id,
        "seg_b": seg_b_id,
        "should_merge": label,
        "n_contacts": n_contacts,
        "contacts": contacts_padded.tolist(),
        "pointcloud_a": _sample_mesh_points(mesh_a_cropped, cfg.n_pointcloud_points).tolist(),
        "pointcloud_b": _sample_mesh_points(mesh_b_cropped, cfg.n_pointcloud_points).tolist(),
        **cfg.metadata,
    }


def _download_meshes_for_pairs(
    mesh_cv: CloudVolume | None,
    segs: set[int],
    idx: VolumetricIndex,
    crop_pad: Sequence[int],
    resolution: np.ndarray,
    coord_str: str,
) -> dict[int, trimesh.Trimesh]:
    """Download and clip meshes for segment IDs."""
    if mesh_cv is None or not segs:
        return {}
    idx_padded = idx.padded(Vec3D[int](*crop_pad))
    bbox_start = (
        np.array([idx_padded.start[0], idx_padded.start[1], idx_padded.start[2]]) * resolution
    )
    bbox_end = np.array([idx_padded.stop[0], idx_padded.stop[1], idx_padded.stop[2]]) * resolution
    meshes = _download_and_clip_meshes(mesh_cv, list(segs), bbox_start, bbox_end)
    print(f"[{coord_str}] Downloaded {len(meshes)} meshes", flush=True)
    return meshes


def _make_sample_config(
    idx: VolumetricIndex,
    crop_pad: Sequence[int],
    candidate_path: str,
    reference_path: str,
    affinity_path: str,
    max_contact_vx: int,
    n_pointcloud_points: int,
    sphere_radius_nm: float,
) -> SampleConfig:
    """Create sample configuration from index and parameters."""
    resolution = np.array([idx.resolution[0], idx.resolution[1], idx.resolution[2]])
    chunk_coord = [int(idx.start[0]), int(idx.start[1]), int(idx.start[2])]
    return SampleConfig(
        resolution=resolution,
        sphere_radius_nm=sphere_radius_nm,
        n_pointcloud_points=n_pointcloud_points,
        max_contact_vx=max_contact_vx,
        coord_str=f"{chunk_coord[0]}_{chunk_coord[1]}_{chunk_coord[2]}",
        metadata={
            "chunk_coord": chunk_coord,
            "chunk_size": [int(idx.stop[i] - idx.start[i]) for i in range(3)],
            "crop_pad": list(crop_pad),
            "sphere_radius_nm": sphere_radius_nm,
            "candidate_path": candidate_path,
            "reference_path": reference_path,
            "affinity_path": affinity_path,
        },
    )


def _generate_all_samples(
    valid_pairs: list[tuple[int, int, int]],
    meshes: dict[int, trimesh.Trimesh],
    contact_data: dict[tuple[int, int], list[tuple[float, float, float, float]]],
    cand_to_proof: dict[int, set[int]],
    cfg: SampleConfig,
) -> list[dict[str, Any]]:
    """Generate samples for all valid pairs."""
    samples, n_no_mesh = [], 0
    for seg_a_id, seg_b_id, _ in valid_pairs:
        sample = _generate_single_sample(
            seg_a_id, seg_b_id, meshes, contact_data, cand_to_proof, cfg
        )
        if sample is None:
            n_no_mesh += 1
        else:
            samples.append(sample)
    print(f"[{cfg.coord_str}]   no mesh: -{n_no_mesh}", flush=True)
    print(f"[{cfg.coord_str}]   final samples: {len(samples)}", flush=True)
    return samples


@contextlib.contextmanager
def _timed(label: str, coord_str: str):
    """Context manager for timing code blocks."""
    t0 = time.time()
    yield
    print(f"[{coord_str}]   {label}: {time.time() - t0:.1f}s", flush=True)


def process_contact_samples(
    candidate_seg: np.ndarray,
    proofread_seg: np.ndarray,
    affinity_raw: np.ndarray,
    mesh_cv: CloudVolume | None,
    idx: VolumetricIndex,
    crop_pad: Sequence[int],
    candidate_path: str,
    reference_path: str,
    affinity_path: str,
    flt: FilterParams,
    n_pointcloud_points: int,
    sphere_radius_nm: float,
) -> list[dict[str, Any]]:
    """Process volumes and produce training samples."""
    cfg = _make_sample_config(
        idx,
        crop_pad,
        candidate_path,
        reference_path,
        affinity_path,
        flt.max_contact_vx,
        n_pointcloud_points,
        sphere_radius_nm,
    )

    with _timed("overlaps/filtering", cfg.coord_str):
        exclude_ids, cand_to_proof = _find_excluded_segments_and_overlaps(
            candidate_seg, proofread_seg, flt.min_seg_size_vx, flt.min_overlap_vx, cfg.coord_str
        )

    with _timed(f"blackout ({len(exclude_ids)} segs)", cfg.coord_str):
        candidate_seg = _blackout_segments(candidate_seg, exclude_ids)

    with _timed("find contacts", cfg.coord_str):
        seg_a, seg_b, aff, cx, cy, cz = _find_contacts(
            candidate_seg, affinity_raw, idx.start, crop_pad
        )
    if len(seg_a) == 0:
        print(f"[{cfg.coord_str}] No contacts found", flush=True)
        return []

    with _timed("filter pairs", cfg.coord_str):
        valid_pairs, segs = _filter_contact_pairs(
            _compute_contact_counts(seg_a, seg_b),
            flt.min_contact_vx,
            flt.max_contact_vx,
            cfg.coord_str,
        )
    if not valid_pairs:
        return []

    with _timed("mesh download", cfg.coord_str):
        meshes = _download_meshes_for_pairs(
            mesh_cv, segs, idx, crop_pad, cfg.resolution, cfg.coord_str
        )

    with _timed("build lookup", cfg.coord_str):
        contact_data = _build_contact_lookup(seg_a, seg_b, cx, cy, cz, aff)

    with _timed("generate samples", cfg.coord_str):
        return _generate_all_samples(valid_pairs, meshes, contact_data, cand_to_proof, cfg)


@builder.register("ContactSampleOp")
@taskable_operation_cls
@attrs.frozen
class ContactSampleOp:
    output_path: str
    sphere_radius_nm: float
    crop_pad: Sequence[int] = (0, 0, 0)
    min_seg_size_vx: int = 2000
    min_overlap_vx: int = 1000
    min_contact_vx: int = 5
    max_contact_vx: int = 2048
    n_pointcloud_points: int = 2048

    def get_input_resolution(self, dst_resolution: Vec3D[float]) -> Vec3D[float]:
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> ContactSampleOp:
        return attrs.evolve(self, crop_pad=Vec3D[int](*self.crop_pad) + crop_pad)

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer | None,
        candidate_layer: VolumetricLayer,
        reference_layer: VolumetricLayer,
        affinity_layer: VolumetricLayer,
    ) -> None:
        del dst
        coord_str = f"{int(idx.start[0])}_{int(idx.start[1])}_{int(idx.start[2])}"
        t_start = time.time()
        print(f"[{coord_str}] Starting...", flush=True)

        idx_padded = idx.padded(Vec3D[int](*self.crop_pad))
        print(f"[{coord_str}] Reading...", flush=True)
        t0 = time.time()
        with semaphore("read"):
            candidate_seg, proofread_seg, affinity_raw = _read_layers_parallel(
                candidate_layer, reference_layer, affinity_layer, idx_padded
            )
        print(f"[{coord_str}] Reading done: {time.time() - t0:.1f}s", flush=True)

        mesh_cv = CloudVolume(candidate_layer.backend.name, use_https=True, progress=False)

        print(f"[{coord_str}] Processing...", flush=True)
        t0 = time.time()
        flt = FilterParams(
            min_seg_size_vx=self.min_seg_size_vx,
            min_overlap_vx=self.min_overlap_vx,
            min_contact_vx=self.min_contact_vx,
            max_contact_vx=self.max_contact_vx,
        )
        samples = process_contact_samples(
            candidate_seg,
            proofread_seg,
            affinity_raw,
            mesh_cv,
            idx_padded,
            self.crop_pad,
            candidate_layer.backend.name,
            reference_layer.backend.name,
            affinity_layer.backend.name,
            flt,
            self.n_pointcloud_points,
            self.sphere_radius_nm,
        )
        print(f"[{coord_str}] Processing done: {time.time() - t0:.1f}s", flush=True)

        print(f"[{coord_str}] Saving...", flush=True)
        t0 = time.time()
        table = pa.Table.from_pylist(samples) if samples else _make_empty_table()
        with semaphore("write"):
            _write_feather(table, f"{self.output_path}/samples_{coord_str}.feather")
        print(f"[{coord_str}] Saved {len(samples)} samples: {time.time() - t0:.1f}s", flush=True)
        print(f"[{coord_str}] Done: {time.time() - t_start:.1f}s total", flush=True)

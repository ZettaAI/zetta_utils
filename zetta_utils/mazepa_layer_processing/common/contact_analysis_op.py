from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import attrs
import cc3d
import fsspec
import numpy as np
import pandas as pd
import trimesh
from cloudvolume import CloudVolume

from zetta_utils import builder
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.mazepa import taskable_operation_cls
from zetta_utils.mazepa.semaphores import semaphore


def _get_edge_segment_ids(seg: np.ndarray) -> np.ndarray:
    edge_ids = np.unique(
        np.concatenate(
            (
                seg[0, :, :].flatten(),
                seg[-1, :, :].flatten(),
                seg[:, 0, :].flatten(),
                seg[:, -1, :].flatten(),
                seg[:, :, 0].flatten(),
                seg[:, :, -1].flatten(),
            )
        )
    )
    edge_ids = edge_ids[edge_ids != 0]
    return edge_ids


def _find_contacts_fast(
    candidate_seg: np.ndarray, affinity_mean: np.ndarray, start: Vec3D
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find all contacts across all 3 axes efficiently."""
    results = []
    sx, sy, sz = int(start[0]), int(start[1]), int(start[2])

    # X-axis contacts
    mask = (
        (candidate_seg[:-1] != candidate_seg[1:])
        & (candidate_seg[:-1] != 0)
        & (candidate_seg[1:] != 0)
    )
    idx = np.nonzero(mask)
    if len(idx[0]) > 0:
        results.append(
            (
                candidate_seg[:-1][mask],
                candidate_seg[1:][mask],
                affinity_mean[:-1][mask],
                idx[0] + sx,
                idx[1] + sy,
                idx[2] + sz,
            )
        )

    # Y-axis contacts
    mask = (
        (candidate_seg[:, :-1] != candidate_seg[:, 1:])
        & (candidate_seg[:, :-1] != 0)
        & (candidate_seg[:, 1:] != 0)
    )
    idx = np.nonzero(mask)
    if len(idx[0]) > 0:
        results.append(
            (
                candidate_seg[:, :-1][mask],
                candidate_seg[:, 1:][mask],
                affinity_mean[:, :-1][mask],
                idx[0] + sx,
                idx[1] + sy,
                idx[2] + sz,
            )
        )

    # Z-axis contacts
    mask = (
        (candidate_seg[:, :, :-1] != candidate_seg[:, :, 1:])
        & (candidate_seg[:, :, :-1] != 0)
        & (candidate_seg[:, :, 1:] != 0)
    )
    idx = np.nonzero(mask)
    if len(idx[0]) > 0:
        results.append(
            (
                candidate_seg[:, :, :-1][mask],
                candidate_seg[:, :, 1:][mask],
                affinity_mean[:, :, :-1][mask],
                idx[0] + sx,
                idx[1] + sy,
                idx[2] + sz,
            )
        )

    if not results:
        empty = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float32)
        return empty, empty, empty_f, empty, empty, empty

    return (
        np.concatenate([r[0] for r in results]),
        np.concatenate([r[1] for r in results]),
        np.concatenate([r[2] for r in results]),
        np.concatenate([r[3] for r in results]),
        np.concatenate([r[4] for r in results]),
        np.concatenate([r[5] for r in results]),
    )


def _compute_overlaps(
    candidate_seg: np.ndarray, proofread_seg: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute overlaps between candidate segments and proofread connected components."""
    cc_proofread = cc3d.connected_components(proofread_seg, connectivity=6)
    flat_candidate = candidate_seg.ravel()
    flat_cc_proof = cc_proofread.ravel()
    valid_mask = (flat_candidate != 0) & (flat_cc_proof != 0)
    df = pd.DataFrame(
        {
            "cand": flat_candidate[valid_mask],
            "cc_proof": flat_cc_proof[valid_mask],
        }
    )
    counts_df = df.groupby(["cand", "cc_proof"]).size().reset_index(name="count")
    return (
        counts_df["cand"].values.astype(np.int64),
        counts_df["cc_proof"].values.astype(np.int64),
        counts_df["count"].values.astype(np.int32),
    )


# pylint: disable=too-many-arguments
def _compute_contact_stats(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    aff: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "seg_a": seg_a,
            "seg_b": seg_b,
            "aff": aff,
            "x": x,
            "y": y,
            "z": z,
        }
    )
    grouped = df.groupby(["seg_a", "seg_b"])
    stats = grouped.agg(
        aff_mean=("aff", "mean"),
        aff_median=("aff", "median"),
        count=("aff", "size"),
        aff_sum=("aff", "sum"),
        x_weighted=("x", lambda v: (v * df.loc[v.index, "aff"]).sum()),
        y_weighted=("y", lambda v: (v * df.loc[v.index, "aff"]).sum()),
        z_weighted=("z", lambda v: (v * df.loc[v.index, "aff"]).sum()),
    ).reset_index()
    stats["com_x"] = stats["x_weighted"] / stats["aff_sum"]
    stats["com_y"] = stats["y_weighted"] / stats["aff_sum"]
    stats["com_z"] = stats["z_weighted"] / stats["aff_sum"]
    return stats[["seg_a", "seg_b", "aff_mean", "aff_median", "count", "com_x", "com_y", "com_z"]]


def _process_meshes_batch(
    seg_ids: list[int],
    meshes: dict,
    box_facets_origin: np.ndarray,
    box_normals: np.ndarray,
    n_points: int,
) -> dict[int, np.ndarray]:
    """Process a batch of meshes: clip and sample."""
    result: dict[int, np.ndarray] = {}

    valid_meshes = []
    valid_ids = []
    for seg_id in seg_ids:
        mesh_obj = meshes.get(seg_id)
        if mesh_obj is None:
            continue
        vertices, faces = mesh_obj.vertices, mesh_obj.faces
        if len(vertices) == 0 or len(faces) == 0:
            continue
        valid_meshes.append(trimesh.Trimesh(vertices=vertices, faces=faces))
        valid_ids.append(seg_id)

    if not valid_meshes:
        return result

    for seg_id, mesh in zip(valid_ids, valid_meshes):
        clipped = mesh.slice_plane(box_facets_origin, -box_normals)
        if clipped is None or len(clipped.vertices) == 0 or len(clipped.faces) == 0:
            continue
        sample_result = trimesh.sample.sample_surface(
            clipped, min(n_points, len(clipped.faces) * 10)
        )
        result[seg_id] = sample_result[0].astype(np.float32)

    return result


def _sample_mesh_surface_points(
    cv: CloudVolume,
    segment_ids: list[int],
    bbox_start: np.ndarray,
    bbox_end: np.ndarray,
    n_points: int = 2000,
) -> dict[int, np.ndarray]:
    """Download meshes, clip to bbox, sample surface points."""
    if not segment_ids:
        return {}

    meshes = cv.mesh.get(segment_ids, progress=True)

    box_center = (bbox_start + bbox_end) / 2
    box_extents = bbox_end - bbox_start
    box = trimesh.creation.box(extents=box_extents)
    box.apply_translation(box_center)
    box_normals = np.array(box.facets_normal)
    box_facets_origin = box.facets_origin

    return _process_meshes_batch(segment_ids, meshes, box_facets_origin, box_normals, n_points)


def _save_npz(path: str, arrays: dict[str, Any]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    buffer.seek(0)
    with fsspec.open(path, "wb") as f:
        f.write(buffer.read())


def _read_layers_parallel(
    candidate: VolumetricLayer,
    proofread: VolumetricLayer,
    affinity: VolumetricLayer,
    idx_padded: VolumetricIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def read_layer(layer: VolumetricLayer, idx: VolumetricIndex) -> np.ndarray:
        return np.asarray(layer[idx])

    with ThreadPoolExecutor(max_workers=3) as executor:
        cand_future = executor.submit(read_layer, candidate, idx_padded)
        proof_future = executor.submit(read_layer, proofread, idx_padded)
        aff_future = executor.submit(read_layer, affinity, idx_padded)
        return cand_future.result().squeeze(), proof_future.result().squeeze(), aff_future.result()


def _prepare_mesh_arrays(
    mesh_points: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if mesh_points:
        seg_ids = np.array(list(mesh_points.keys()), dtype=np.int64)
        all_points = np.concatenate(list(mesh_points.values()))
        offsets = np.cumsum([0] + [len(pts) for pts in mesh_points.values()][:-1]).astype(np.int32)
        counts = np.array([len(pts) for pts in mesh_points.values()], dtype=np.int32)
    else:
        seg_ids = np.array([], dtype=np.int64)
        all_points = np.array([], dtype=np.float32).reshape(0, 3)
        offsets = np.array([], dtype=np.int32)
        counts = np.array([], dtype=np.int32)
    return seg_ids, all_points, offsets, counts


# pylint: disable=too-many-arguments
def _build_chunk_data(  # noqa: PLR0913
    idx: VolumetricIndex,
    crop_pad: Sequence[int],
    seg_lo: np.ndarray,
    seg_hi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    aff: np.ndarray,
    contact_stats: pd.DataFrame,
    overlap_cand: np.ndarray,
    overlap_proof: np.ndarray,
    overlap_count: np.ndarray,
    edge_ids: np.ndarray,
    mesh_seg_ids: np.ndarray,
    mesh_all_points: np.ndarray,
    mesh_offsets: np.ndarray,
    mesh_counts: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "chunk_start": np.array([idx.start[0], idx.start[1], idx.start[2]], dtype=np.int64),
        "chunk_end": np.array([idx.stop[0], idx.stop[1], idx.stop[2]], dtype=np.int64),
        "resolution": np.array(
            [idx.resolution[0], idx.resolution[1], idx.resolution[2]], dtype=np.float32
        ),
        "crop_pad": np.array(crop_pad, dtype=np.int32),
        "contacts_seg_a": seg_lo.astype(np.int64),
        "contacts_seg_b": seg_hi.astype(np.int64),
        "contacts_x": x.astype(np.int32),
        "contacts_y": y.astype(np.int32),
        "contacts_z": z.astype(np.int32),
        "contacts_aff": aff.astype(np.float32),
        "stats_seg_a": contact_stats["seg_a"].values.astype(np.int64),
        "stats_seg_b": contact_stats["seg_b"].values.astype(np.int64),
        "stats_count": contact_stats["count"].values.astype(np.int32),
        "stats_aff_mean": contact_stats["aff_mean"].values.astype(np.float32),
        "stats_aff_median": contact_stats["aff_median"].values.astype(np.float32),
        "stats_com_x": contact_stats["com_x"].values.astype(np.float32),
        "stats_com_y": contact_stats["com_y"].values.astype(np.float32),
        "stats_com_z": contact_stats["com_z"].values.astype(np.float32),
        "overlaps_cand_id": overlap_cand,
        "overlaps_proof_cc": overlap_proof,
        "overlaps_count": overlap_count,
        "edge_ids": edge_ids.astype(np.int64),
        "mesh_seg_ids": mesh_seg_ids,
        "mesh_points": mesh_all_points,
        "mesh_offsets": mesh_offsets,
        "mesh_counts": mesh_counts,
    }


def _find_all_contacts(
    candidate_seg: np.ndarray,
    affinity_mean: np.ndarray,
    start: Vec3D,
    crop_pad: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seg_a, seg_b, aff, x, y, z = _find_contacts_fast(candidate_seg, affinity_mean, start)

    swap = seg_a > seg_b
    seg_lo, seg_hi = np.where(swap, seg_b, seg_a), np.where(swap, seg_a, seg_b)

    # Filter to keep only contacts within the inner (unpadded) kernel region
    shape = candidate_seg.shape
    kernel_start = np.array([start[0], start[1], start[2]]) + np.array(crop_pad)
    kernel_end = np.array([start[0], start[1], start[2]]) + np.array(shape) - np.array(crop_pad)
    in_kernel = (
        (x >= kernel_start[0])
        & (x < kernel_end[0])
        & (y >= kernel_start[1])
        & (y < kernel_end[1])
        & (z >= kernel_start[2])
        & (z < kernel_end[2])
    )

    return (
        seg_lo[in_kernel],
        seg_hi[in_kernel],
        aff[in_kernel],
        x[in_kernel],
        y[in_kernel],
        z[in_kernel],
    )


@builder.register("ContactAnalysisOp")
@taskable_operation_cls
@attrs.frozen
class ContactAnalysisOp:
    output_path: str
    mesh_path: str | None = None
    n_surface_points: int = 2000
    crop_pad: Sequence[int] = (0, 0, 0)

    def get_input_resolution(self, dst_resolution: Vec3D[float]) -> Vec3D[float]:
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> ContactAnalysisOp:
        return attrs.evolve(self, crop_pad=Vec3D[int](*self.crop_pad) + crop_pad)

    # pylint: disable=too-many-locals
    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer | None,
        candidate: VolumetricLayer,
        proofread: VolumetricLayer,
        affinity: VolumetricLayer,
    ) -> None:
        coord_str = f"{int(idx.start[0])}_{int(idx.start[1])}_{int(idx.start[2])}"
        t_start = time.time()
        print(f"[{coord_str}] Starting...", flush=True)

        idx_padded = idx.padded(Vec3D[int](*self.crop_pad))
        print(f"[{coord_str}] Reading...", flush=True)
        t0 = time.time()
        with semaphore("read"):
            candidate_seg, proofread_seg, affinity_raw = _read_layers_parallel(
                candidate, proofread, affinity, idx_padded
            )
        print(f"[{coord_str}] Reading done: {time.time() - t0:.1f}s", flush=True)

        print(f"[{coord_str}] Processing...", flush=True)
        t0 = time.time()

        t1 = time.time()
        affinity_mean = np.mean(affinity_raw, axis=0)
        print(f"[{coord_str}]   np.mean: {time.time() - t1:.2f}s", flush=True)
        del affinity_raw

        t1 = time.time()
        overlap_cand, overlap_proof, overlap_count = _compute_overlaps(
            candidate_seg, proofread_seg
        )
        print(f"[{coord_str}]   overlaps: {time.time() - t1:.2f}s", flush=True)
        del proofread_seg

        t1 = time.time()
        edge_ids = _get_edge_segment_ids(candidate_seg)
        print(f"[{coord_str}]   edge_ids: {time.time() - t1:.2f}s", flush=True)

        print(f"[{coord_str}] Processing done: {time.time() - t0:.1f}s", flush=True)

        print(f"[{coord_str}] Finding contacts...", flush=True)
        t0 = time.time()
        seg_lo, seg_hi, aff, x, y, z = _find_all_contacts(
            candidate_seg, affinity_mean, idx_padded.start, self.crop_pad
        )
        del candidate_seg, affinity_mean
        contact_stats = _compute_contact_stats(seg_lo, seg_hi, aff, x, y, z)
        n_pairs = len(contact_stats)
        print(
            f"[{coord_str}] Contacts done: {time.time() - t0:.1f}s ({n_pairs} pairs)", flush=True
        )

        all_contact_segs = list(set(np.unique(seg_lo).tolist() + np.unique(seg_hi).tolist()))

        print(f"[{coord_str}] Downloading meshes ({len(all_contact_segs)} segs)...", flush=True)
        t0 = time.time()
        mesh_points = self._sample_meshes_for_segs(idx, all_contact_segs)
        n_meshes = len(mesh_points)
        print(f"[{coord_str}] Meshes done: {time.time() - t0:.1f}s ({n_meshes} segs)", flush=True)

        mesh_seg_ids, mesh_all_points, mesh_offsets, mesh_counts = _prepare_mesh_arrays(
            mesh_points
        )

        print(f"[{coord_str}] Saving...", flush=True)
        t0 = time.time()
        chunk_data = _build_chunk_data(
            idx,
            self.crop_pad,
            seg_lo,
            seg_hi,
            x,
            y,
            z,
            aff,
            contact_stats,
            overlap_cand,
            overlap_proof,
            overlap_count,
            edge_ids,
            mesh_seg_ids,
            mesh_all_points,
            mesh_offsets,
            mesh_counts,
        )
        with semaphore("write"):
            _save_npz(f"{self.output_path}/chunk_{coord_str}.npz", chunk_data)
        print(f"[{coord_str}] Saving done: {time.time() - t0:.1f}s", flush=True)
        print(f"[{coord_str}] Done: {time.time() - t_start:.1f}s total", flush=True)

    def _sample_meshes_for_segs(
        self, idx: VolumetricIndex, segment_ids: list[int]
    ) -> dict[int, np.ndarray]:
        if self.mesh_path is None or not segment_ids:
            return {}
        cv = CloudVolume(self.mesh_path, use_https=True, progress=False)
        resolution = idx.resolution
        idx_padded = idx.padded(Vec3D[int](*self.crop_pad))
        bbox_start = np.array(
            [idx_padded.start[0], idx_padded.start[1], idx_padded.start[2]]
        ) * np.array([resolution[0], resolution[1], resolution[2]])
        bbox_end = np.array(
            [idx_padded.stop[0], idx_padded.stop[1], idx_padded.stop[2]]
        ) * np.array([resolution[0], resolution[1], resolution[2]])
        return _sample_mesh_surface_points(
            cv, segment_ids, bbox_start, bbox_end, self.n_surface_points
        )

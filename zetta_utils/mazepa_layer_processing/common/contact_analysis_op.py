from __future__ import annotations

import io
from typing import Any, Sequence

import attrs
import cc3d
import fsspec
import numpy as np
import pandas as pd

from zetta_utils import builder
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.mazepa import taskable_operation_cls
from zetta_utils.mazepa.semaphores import semaphore


def _get_edge_segment_ids(seg: np.ndarray) -> set[int]:
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
    return set(edge_ids)


def _find_axis_contacts(
    candidate_seg: np.ndarray, affinity_mean: np.ndarray, axis: int, start: Vec3D
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = candidate_seg.shape
    slices_a = [slice(None), slice(None), slice(None)]
    slices_b = [slice(None), slice(None), slice(None)]
    slices_a[axis] = slice(None, -1)
    slices_b[axis] = slice(1, None)

    seg_a = candidate_seg[tuple(slices_a)].ravel()
    seg_b = candidate_seg[tuple(slices_b)].ravel()
    aff_vals = affinity_mean[tuple(slices_a)].ravel()

    ranges = [np.arange(s) for s in shape]
    ranges[axis] = np.arange(shape[axis] - 1)
    x, y, z = np.meshgrid(*ranges, indexing="ij")

    boundary_mask = (seg_a != seg_b) & (seg_a != 0) & (seg_b != 0)
    return (
        seg_a[boundary_mask],
        seg_b[boundary_mask],
        aff_vals[boundary_mask],
        x.ravel()[boundary_mask] + int(start[0]),
        y.ravel()[boundary_mask] + int(start[1]),
        z.ravel()[boundary_mask] + int(start[2]),
    )


def _compute_overlaps(candidate_seg: np.ndarray, proofread_seg: np.ndarray) -> pd.DataFrame:
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
    return df.groupby(["cand", "cc_proof"]).size().reset_index(name="count")


def _save_npz(path: str, arrays: dict[str, Any]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    buffer.seek(0)
    with fsspec.open(path, "wb") as f:
        f.write(buffer.read())


@builder.register("ContactAnalysisOp")
@taskable_operation_cls
@attrs.frozen
class ContactAnalysisOp:
    output_path: str
    min_contact_vx: int = 10
    crop_pad: Sequence[int] = (0, 0, 0)

    def get_input_resolution(self, dst_resolution: Vec3D[float]) -> Vec3D[float]:
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> ContactAnalysisOp:
        return attrs.evolve(self, crop_pad=Vec3D[int](*self.crop_pad) + crop_pad)

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer | None,
        candidate: VolumetricLayer,
        proofread: VolumetricLayer,
        affinity: VolumetricLayer,
    ) -> None:
        with semaphore("read"):
            candidate_seg = np.asarray(candidate[idx]).squeeze()
            proofread_seg = np.asarray(proofread[idx]).squeeze()
            affinity_raw = np.asarray(affinity[idx])

        affinity_mean = np.mean(affinity_raw, axis=0)
        counts_df = _compute_overlaps(candidate_seg, proofread_seg)
        edge_ids = _get_edge_segment_ids(candidate_seg)

        contacts = [
            _find_axis_contacts(candidate_seg, affinity_mean, ax, idx.start) for ax in range(3)
        ]
        seg_a = np.concatenate([c[0] for c in contacts])
        seg_b = np.concatenate([c[1] for c in contacts])
        aff = np.concatenate([c[2] for c in contacts])
        x = np.concatenate([c[3] for c in contacts])
        y = np.concatenate([c[4] for c in contacts])
        z = np.concatenate([c[5] for c in contacts])

        swap = seg_a > seg_b
        seg_lo, seg_hi = np.where(swap, seg_b, seg_a), np.where(swap, seg_a, seg_b)
        keep = ~(np.isin(seg_lo, list(edge_ids)) | np.isin(seg_hi, list(edge_ids)))
        seg_lo, seg_hi, aff, x, y, z = (
            seg_lo[keep],
            seg_hi[keep],
            aff[keep],
            x[keep],
            y[keep],
            z[keep],
        )

        coord_str = f"{int(idx.start[0])}_{int(idx.start[1])}_{int(idx.start[2])}"
        with semaphore("write"):
            _save_npz(
                f"{self.output_path}/contacts_{coord_str}.npz",
                {
                    "seg_a": seg_lo.astype(np.int32),
                    "seg_b": seg_hi.astype(np.int32),
                    "x": x.astype(np.int32),
                    "y": y.astype(np.int32),
                    "z": z.astype(np.int32),
                    "aff": aff.astype(np.float32),
                },
            )
            _save_npz(
                f"{self.output_path}/overlaps_{coord_str}.npz",
                {
                    "cand_id": counts_df["cand"].values.astype(np.int64),
                    "cc_proof_id": counts_df["cc_proof"].values.astype(np.int64),
                    "count": counts_df["count"].values.astype(np.int32),
                },
            )

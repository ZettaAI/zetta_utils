from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import attrs
import numpy as np

from zetta_utils import builder
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.layer.volumetric.seg_contact import SegContact, VolumetricSegContactLayer
from zetta_utils.mazepa import taskable_operation_cls
from zetta_utils.mazepa.semaphores import semaphore


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


def _find_contacts(
    seg: np.ndarray, aff: np.ndarray, start: Vec3D
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find contacts between segments using affinity data."""
    sx, sy, sz = float(start[0]), float(start[1]), float(start[2])
    results = []

    # For each axis, offset by 0.5 on the contact axis to get face center
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

    # Normalize segment order so seg_a < seg_b
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
    # Find pairs with any contact outside kernel
    pairs_outside: set[tuple[int, int]] = set()
    for a, b, out in zip(seg_a, seg_b, outside_kernel):
        if out:
            pairs_outside.add((int(a), int(b)))
    # Keep only contacts from pairs fully inside kernel
    keep = np.array([(int(a), int(b)) not in pairs_outside for a, b in zip(seg_a, seg_b)])
    return seg_a[keep], seg_b[keep], aff[keep], x[keep], y[keep], z[keep]


def _build_seg_contacts(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    aff: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    resolution: Vec3D,
    min_contact_vx: int,
    max_contact_vx: int,
) -> list[SegContact]:
    """Build SegContact objects from contact data."""
    # Group contacts by segment pair
    pair_data: dict[tuple[int, int], list[tuple[float, float, float, float]]] = defaultdict(list)
    for a, b, af, cx, cy, cz in zip(seg_a, seg_b, aff, x, y, z):
        pair_data[(int(a), int(b))].append((float(cx), float(cy), float(cz), float(af)))

    contacts = []
    contact_id = 0
    for (a, b), faces in pair_data.items():
        n_faces = len(faces)
        if n_faces < min_contact_vx or n_faces > max_contact_vx:
            continue

        # Convert to nm and compute COM (affinity-weighted)
        faces_arr = np.array(faces, dtype=np.float32)
        x_nm = faces_arr[:, 0] * resolution[0]
        y_nm = faces_arr[:, 1] * resolution[1]
        z_nm = faces_arr[:, 2] * resolution[2]
        aff_vals = faces_arr[:, 3]

        aff_sum = aff_vals.sum()
        if aff_sum > 0:
            com_x = (x_nm * aff_vals).sum() / aff_sum
            com_y = (y_nm * aff_vals).sum() / aff_sum
            com_z = (z_nm * aff_vals).sum() / aff_sum
        else:
            com_x, com_y, com_z = x_nm.mean(), y_nm.mean(), z_nm.mean()

        # Build contact_faces array: (N, 4) with x, y, z, affinity in nm
        contact_faces = np.column_stack([x_nm, y_nm, z_nm, aff_vals]).astype(np.float32)

        contacts.append(
            SegContact(
                id=contact_id,
                seg_a=a,
                seg_b=b,
                com=Vec3D(float(com_x), float(com_y), float(com_z)),
                contact_faces=contact_faces,
            )
        )
        contact_id += 1

    return contacts


@builder.register("SegContactOp")
@taskable_operation_cls
@attrs.frozen
class SegContactOp:
    """Operation to find and write segment contacts."""

    crop_pad: Sequence[int] = (0, 0, 0)
    min_contact_vx: int = 5
    max_contact_vx: int = 2048

    def get_input_resolution(self, dst_resolution: Vec3D[float]) -> Vec3D[float]:
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> SegContactOp:
        return attrs.evolve(self, crop_pad=Vec3D[int](*self.crop_pad) + crop_pad)

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricSegContactLayer,
        segmentation_layer: VolumetricLayer,
        affinity_layer: VolumetricLayer,
    ) -> None:
        idx_padded = idx.padded(Vec3D[int](*self.crop_pad))

        with semaphore("read"):
            seg = np.asarray(segmentation_layer[idx_padded]).squeeze()
            aff = np.asarray(affinity_layer[idx_padded])

        # Find all contacts
        seg_a, seg_b, aff_vals, x, y, z = _find_contacts(seg, aff, idx_padded.start)

        if len(seg_a) == 0:
            return

        # Filter to kernel region
        seg_a, seg_b, aff_vals, x, y, z = _filter_pairs_to_kernel(
            seg_a, seg_b, aff_vals, x, y, z, idx_padded.start, seg.shape, self.crop_pad
        )

        if len(seg_a) == 0:
            return

        # Build SegContact objects
        contacts = _build_seg_contacts(
            seg_a,
            seg_b,
            aff_vals,
            x,
            y,
            z,
            idx.resolution,
            self.min_contact_vx,
            self.max_contact_vx,
        )

        if contacts:
            with semaphore("write"):
                dst[idx] = contacts

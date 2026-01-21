from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
from scipy.spatial.transform import Rotation

from zetta_utils import builder
from zetta_utils.geometry import Vec3D

from .contact import SegContact


def _random_rotation_matrix() -> np.ndarray:
    """Generate a random 3D rotation matrix."""
    return Rotation.random().as_matrix().astype(np.float32)


def resample_points(
    points: np.ndarray,
    target_n: int,
    weighting: str = "uniform",
    center: np.ndarray | None = None,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Resample points to target count with optional distance-based weighting.

    Args:
        points: Input points array [N, D] where D >= 3 (first 3 are xyz).
        target_n: Target number of points.
        weighting: Weighting scheme for sampling probability:
            - "uniform": Equal probability for all points
            - "inverse_r": Weight proportional to 1/r (favor points near center)
            - "inverse_r2": Weight proportional to 1/r² (stronger center bias)
        center: Center point for distance calculation [3]. Defaults to origin.
        return_indices: If True, also return the indices used for sampling.

    Returns:
        If return_indices=False: Resampled points [target_n, D].
        If return_indices=True: Tuple of (resampled points, indices).
        Returns input unchanged (no-op) when target_n == n_points and weighting == "uniform".
    """
    n_points = points.shape[0]

    # Handle empty input
    if n_points == 0:
        result = np.zeros((target_n, points.shape[1]), dtype=points.dtype)
        if return_indices:
            return result, np.zeros(target_n, dtype=np.int64)
        return result

    # Handle zero target
    if target_n == 0:
        result = np.zeros((0, points.shape[1]), dtype=points.dtype)
        if return_indices:
            return result, np.zeros(0, dtype=np.int64)
        return result

    # True no-op: same count with uniform weighting
    if target_n == n_points and weighting == "uniform":
        if return_indices:
            return points, np.arange(n_points, dtype=np.int64)
        return points

    if center is None:
        center = np.zeros(3, dtype=np.float32)

    # Compute distances from center
    distances = np.linalg.norm(points[:, :3] - center, axis=1)

    # Compute sampling weights
    if weighting == "uniform":
        weights = np.ones(n_points, dtype=np.float32)
    elif weighting == "inverse_r":
        weights = 1.0 / (distances + 1e-6)
    elif weighting == "inverse_r2":
        weights = 1.0 / (distances**2 + 1e-6)
    else:
        raise ValueError(f"Unknown weighting: {weighting}. Use 'uniform', 'inverse_r', or 'inverse_r2'.")

    # Normalize to probabilities
    probs = weights / weights.sum()

    # Sample indices (with replacement if upsampling)
    replace = target_n > n_points
    indices = np.random.choice(n_points, size=target_n, replace=replace, p=probs)

    result = points[indices].astype(points.dtype)
    if return_indices:
        return result, indices
    return result


@builder.register("resample_pointclouds")
def resample_pointclouds(
    contacts: Sequence[SegContact],
    segment_target: int | None = None,
    segment_weighting: str = "uniform",
    contact_face_target: int | None = None,
    contact_face_weighting: str = "uniform",
) -> Sequence[SegContact]:
    """Resample pointclouds and contact faces individually to fixed counts.

    Resamples each segment's pointcloud and/or contact faces independently.
    Applied to each (radius, n_points) config in local_pointclouds.

    Args:
        contacts: Sequence of SegContact objects.
        segment_target: Target points per segment. If None, no segment resampling.
        segment_weighting: Weighting for segment resampling.
        contact_face_target: Target points for contact faces. If None, no resampling.
        contact_face_weighting: Weighting for contact face resampling.

    Example CUE usage:
        {"@type": "resample_pointclouds", "@mode": "partial",
         segment_target: 1024, contact_face_target: 256}
    """
    result = []
    for contact in contacts:
        new_local_pointclouds = contact.local_pointclouds
        new_contact_faces = contact.contact_faces

        # Resample segments
        if segment_target is not None and contact.local_pointclouds is not None:
            new_local_pointclouds = {}
            for config_tuple, pc_data in contact.local_pointclouds.items():
                new_local_pointclouds[config_tuple] = {
                    seg_id: resample_points(points, segment_target, segment_weighting)
                    for seg_id, points in pc_data.items()
                }

        # Resample contact faces (filter out zero-padded first)
        if contact_face_target is not None:
            valid_mask = np.any(contact.contact_faces[:, :3] != 0, axis=1)
            valid_cf = contact.contact_faces[valid_mask]
            new_contact_faces = resample_points(valid_cf, contact_face_target, contact_face_weighting)

        result.append(
            SegContact(
                id=contact.id,
                seg_a=contact.seg_a,
                seg_b=contact.seg_b,
                com=contact.com,
                contact_faces=new_contact_faces,
                representative_points=contact.representative_points,
                local_pointclouds=new_local_pointclouds,
                merge_decisions=contact.merge_decisions,
                partner_metadata=contact.partner_metadata,
                contact_faces_original_nm=contact.contact_faces_original_nm,
            )
        )

    return result


@builder.register("resample_combined_pointcloud")
def resample_combined_pointcloud(
    contacts: Sequence[SegContact],
    total_target: int = 4096,
    include_contact_faces: bool = False,
    max_contact_face_fraction: float = 0.1,
    weighting: str = "uniform",
    joint_segment_sampling: bool = False,
) -> Sequence[SegContact]:
    """Allocate total point budget across segments and contact faces, then resample.

    Distributes total_target points between seg_a, seg_b, and optionally contact
    faces. Contact faces get at most max_contact_face_fraction of total points,
    capped by the actual number of valid contact faces available. Remaining
    budget is split evenly between segments (or sampled jointly if enabled).

    Applied to each (radius, n_points) config in local_pointclouds separately.

    Args:
        contacts: Sequence of SegContact objects.
        total_target: Total point budget across all components.
        include_contact_faces: Whether to include contact faces in the budget.
        max_contact_face_fraction: Maximum fraction of budget for contact faces.
            Actual allocation may be less if fewer valid contact faces exist.
        weighting: Weighting for resampling. Options:
            - "uniform", "inverse_r", "inverse_r2": Standard distance-based weighting
            - "balanced", "balanced_inverse_r", "balanced_inverse_r2": Balanced weighting
              that equalizes segment representation within each distance shell. Points
              are downweighted by their segment's local density, so both segments
              contribute equally at each distance where both are present.
        joint_segment_sampling: If True, pool seg_a and seg_b points together and
            sample from the combined pool. Required for balanced weighting modes.

    Example CUE usage:
        {"@type": "resample_combined_pointcloud", "@mode": "partial",
         total_target: 4096, include_contact_faces: true, max_contact_face_fraction: 0.1}
    """
    # Extract base weighting for non-balanced operations (contact faces, independent sampling)
    if weighting.startswith("balanced"):
        base_weighting = weighting[9:] if len(weighting) > 8 else "uniform"
        if base_weighting == "":
            base_weighting = "uniform"
    else:
        base_weighting = weighting

    result = []
    for contact in contacts:
        new_local_pointclouds = contact.local_pointclouds
        new_contact_faces = contact.contact_faces

        if contact.local_pointclouds is None:
            result.append(contact)
            continue

        # Get valid contact faces (filter zero-padded)
        valid_cf_mask = np.any(contact.contact_faces[:, :3] != 0, axis=1)
        valid_cf = contact.contact_faces[valid_cf_mask]
        n_valid_cf = valid_cf.shape[0]

        # Calculate budget allocation
        if include_contact_faces and n_valid_cf > 0:
            max_cf_budget = int(total_target * max_contact_face_fraction)
            cf_budget = min(max_cf_budget, n_valid_cf)
        else:
            cf_budget = 0

        segment_budget_total = total_target - cf_budget

        # Resample segments with allocated budgets
        new_local_pointclouds = {}
        for config_tuple, pc_data in contact.local_pointclouds.items():
            if joint_segment_sampling:
                # Pool all segment points together, sample jointly, then separate
                seg_a_pts = pc_data.get(contact.seg_a, np.zeros((0, 3), dtype=np.float32))
                seg_b_pts = pc_data.get(contact.seg_b, np.zeros((0, 3), dtype=np.float32))
                n_a = seg_a_pts.shape[0]
                n_b = seg_b_pts.shape[0]

                if n_a + n_b > 0:
                    # Combine with segment identity marker
                    combined = np.vstack([seg_a_pts, seg_b_pts])
                    seg_labels = np.array([contact.seg_a] * n_a + [contact.seg_b] * n_b)
                    is_seg_a = seg_labels == contact.seg_a

                    # Check for balanced weighting modes
                    if weighting.startswith("balanced"):
                        # Compute distances
                        distances = np.linalg.norm(combined[:, :3], axis=1)
                        max_dist = distances.max() + 1e-6

                        # Bin points by distance
                        n_bins = 50
                        bin_indices = (distances / max_dist * n_bins).astype(int)
                        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

                        # Compute base distance weights
                        if base_weighting == "uniform":
                            f_r = np.ones(len(combined), dtype=np.float32)
                        elif base_weighting == "inverse_r":
                            f_r = 1.0 / (distances + 1e-6)
                        elif base_weighting == "inverse_r2":
                            f_r = 1.0 / (distances**2 + 1e-6)
                        else:
                            f_r = np.ones(len(combined), dtype=np.float32)

                        # Sum of f(r) per-segment per-bin (for true balancing)
                        sum_f_a = np.zeros(n_bins, dtype=np.float32)
                        sum_f_b = np.zeros(n_bins, dtype=np.float32)
                        for bin_idx in range(n_bins):
                            bin_mask = bin_indices == bin_idx
                            sum_f_a[bin_idx] = f_r[bin_mask & is_seg_a].sum()
                            sum_f_b[bin_idx] = f_r[bin_mask & ~is_seg_a].sum()

                        # Compute balanced weights (vectorized)
                        # Normalize by sum of f(r) for own segment in bin -> equal weight per segment per bin
                        sum_f_self = np.where(is_seg_a, sum_f_a[bin_indices], sum_f_b[bin_indices])
                        weights = f_r / np.maximum(sum_f_self, 1e-6)

                        # Normalize and sample
                        probs = weights / weights.sum()
                        replace = segment_budget_total > len(combined)
                        indices = np.random.choice(len(combined), size=segment_budget_total, replace=replace, p=probs)
                        sampled_points = combined[indices]
                    else:
                        # Standard weighting: use resample_points
                        sampled_points, indices = resample_points(
                            combined, segment_budget_total, weighting, return_indices=True
                        )

                    # Separate back by segment identity
                    sampled_labels = seg_labels[indices]
                    mask_a = sampled_labels == contact.seg_a
                    mask_b = sampled_labels == contact.seg_b

                    new_pc_data = {
                        contact.seg_a: sampled_points[mask_a].astype(np.float32),
                        contact.seg_b: sampled_points[mask_b].astype(np.float32),
                    }
                else:
                    new_pc_data = {
                        contact.seg_a: np.zeros((0, 3), dtype=np.float32),
                        contact.seg_b: np.zeros((0, 3), dtype=np.float32),
                    }
            else:
                # Independent sampling: split budget evenly between segments
                segment_budget_a = segment_budget_total // 2
                segment_budget_b = segment_budget_total - segment_budget_a  # gets remainder if odd

                new_pc_data = {}
                for seg_id, points in pc_data.items():
                    if seg_id == contact.seg_a:
                        new_pc_data[seg_id] = resample_points(points, segment_budget_a, base_weighting)
                    else:
                        new_pc_data[seg_id] = resample_points(points, segment_budget_b, base_weighting)

            new_local_pointclouds[config_tuple] = new_pc_data

        # Resample contact faces if budget allocated (use base_weighting, not balanced)
        if cf_budget > 0:
            new_contact_faces = resample_points(valid_cf, cf_budget, base_weighting)

        # Verify total count
        actual_total = segment_budget_total + cf_budget
        assert actual_total == total_target, (
            f"Point count mismatch: expected {total_target}, got {actual_total} "
            f"(segments={segment_budget_total}, cf={cf_budget})"
        )

        result.append(
            SegContact(
                id=contact.id,
                seg_a=contact.seg_a,
                seg_b=contact.seg_b,
                com=contact.com,
                contact_faces=new_contact_faces,
                representative_points=contact.representative_points,
                local_pointclouds=new_local_pointclouds,
                merge_decisions=contact.merge_decisions,
                partner_metadata=contact.partner_metadata,
                contact_faces_original_nm=contact.contact_faces_original_nm,
            )
        )

    return result


@builder.register("deduplicate_pointclouds")
def deduplicate_pointclouds(
    contacts: Sequence[SegContact],
    apply_to_contact_faces: bool = False,
) -> Sequence[SegContact]:
    """Remove duplicate points from pointclouds and contact faces.

    Should be run before normalization when points are still in integer voxel
    coordinates, so exact comparison works reliably.

    Args:
        contacts: Sequence of SegContact objects.
        apply_to_contact_faces: If True, also deduplicate contact_faces.
    """
    result = []
    for contact in contacts:
        new_local_pointclouds = None
        if contact.local_pointclouds is not None:
            new_local_pointclouds = {}
            for config_tuple, pc_data in contact.local_pointclouds.items():
                new_local_pointclouds[config_tuple] = {
                    seg_id: np.unique(points, axis=0) for seg_id, points in pc_data.items()
                }

        new_contact_faces = contact.contact_faces
        if apply_to_contact_faces and contact.contact_faces.shape[0] > 0:
            new_contact_faces = np.unique(contact.contact_faces, axis=0)

        result.append(
            SegContact(
                id=contact.id,
                seg_a=contact.seg_a,
                seg_b=contact.seg_b,
                com=contact.com,
                contact_faces=new_contact_faces,
                representative_points=contact.representative_points,
                local_pointclouds=new_local_pointclouds,
                merge_decisions=contact.merge_decisions,
                partner_metadata=contact.partner_metadata,
                contact_faces_original_nm=contact.contact_faces_original_nm,
            )
        )

    return result


@builder.register("normalize_pointclouds")
def normalize_pointclouds(
    contacts: Sequence[SegContact],
    normalization_radius_nm: float = 8000.0,
    center_on_random_contact_point: bool = False,
    use_pointcloud_radius: bool = False,
    apply_to_contact_faces: bool = True,
) -> Sequence[SegContact]:
    """Normalize all pointclouds and contact_faces to [-1, 1] centered on COM.

    Args:
        contacts: Sequence of SegContact objects to normalize.
        normalization_radius_nm: Radius in nm for normalization (default 8000).
        center_on_random_contact_point: If True, center on a random point from
            contact_faces instead of the COM.
        use_pointcloud_radius: If True, use the largest local_pointclouds config's
            radius instead of normalization_radius_nm (when local_pointclouds exist).
        apply_to_contact_faces: If True, also normalize contact_faces coordinates.
    """
    result = []
    for contact in contacts:
        if center_on_random_contact_point and contact.contact_faces.shape[0] > 0:
            random_idx = np.random.randint(0, contact.contact_faces.shape[0])
            center = contact.contact_faces[random_idx, :3].astype(np.float32)
        else:
            center = np.array(contact.com, dtype=np.float32)

        # Preserve original contact_faces before normalization
        contact_faces_original_nm = contact.contact_faces.copy()

        # Determine normalization radius for contact_faces
        if use_pointcloud_radius and contact.local_pointclouds:
            largest_config = sorted(contact.local_pointclouds.keys())[-1]
            contact_faces_radius = float(largest_config[0])
        else:
            contact_faces_radius = normalization_radius_nm

        # Normalize contact_faces
        if apply_to_contact_faces:
            new_contact_faces = contact.contact_faces.copy()
            new_contact_faces[:, :3] = (contact.contact_faces[:, :3] - center) / contact_faces_radius
        else:
            new_contact_faces = contact.contact_faces

        # Normalize local_pointclouds if present
        new_local_pointclouds = None
        if contact.local_pointclouds is not None:
            new_local_pointclouds = {}
            for config_tuple, pc_data in contact.local_pointclouds.items():
                # Use unified radius when use_pointcloud_radius=True, else each config's own
                pc_radius = contact_faces_radius if use_pointcloud_radius else float(config_tuple[0])
                new_local_pointclouds[config_tuple] = {
                    seg_id: (points - center) / pc_radius for seg_id, points in pc_data.items()
                }

        result.append(
            SegContact(
                id=contact.id,
                seg_a=contact.seg_a,
                seg_b=contact.seg_b,
                com=Vec3D(*center),
                contact_faces=new_contact_faces,
                representative_points=contact.representative_points,
                local_pointclouds=new_local_pointclouds,
                merge_decisions=contact.merge_decisions,
                partner_metadata=contact.partner_metadata,
                contact_faces_original_nm=contact_faces_original_nm,
            )
        )

    return result


def _add_gaussian_noise_impl(
    contacts: Sequence[SegContact], std: float, apply_to_contact_faces: bool = True
) -> Sequence[SegContact]:
    """Implementation for adding Gaussian noise."""
    result = []
    for contact in contacts:
        # Apply noise to local_pointclouds
        new_local_pointclouds = None
        if contact.local_pointclouds is not None:
            new_local_pointclouds = {}
            for config_tuple, pc_data in contact.local_pointclouds.items():
                new_local_pointclouds[config_tuple] = {
                    seg_id: points + np.random.normal(0, std, points.shape).astype(np.float32)
                    for seg_id, points in pc_data.items()
                }

        # Apply noise to contact_faces
        if apply_to_contact_faces and contact.contact_faces.shape[0] > 0:
            new_contact_faces = contact.contact_faces.copy()
            noise = np.random.normal(0, std, contact.contact_faces[:, :3].shape).astype(np.float32)
            new_contact_faces[:, :3] = contact.contact_faces[:, :3] + noise
        else:
            new_contact_faces = contact.contact_faces

        result.append(
            SegContact(
                id=contact.id,
                seg_a=contact.seg_a,
                seg_b=contact.seg_b,
                com=contact.com,
                contact_faces=new_contact_faces,
                representative_points=contact.representative_points,
                local_pointclouds=new_local_pointclouds if new_local_pointclouds else contact.local_pointclouds,
                merge_decisions=contact.merge_decisions,
                partner_metadata=contact.partner_metadata,
                contact_faces_original_nm=contact.contact_faces_original_nm,
            )
        )

    return result


@builder.register("add_gaussian_noise")
def add_gaussian_noise(
    contacts_or_std: Sequence[SegContact] | float = 0.001,
    std: float = 0.001,
    apply_to_contact_faces: bool = True,
) -> Sequence[SegContact] | Callable:
    """Add Gaussian noise to pointcloud coordinates.

    Can be called two ways:
    - add_gaussian_noise(contacts) - uses default std=0.001
    - add_gaussian_noise(std=0.01) - returns a partial for use as processor

    Args:
        contacts_or_std: Contacts to process, or std value for partial.
        std: Standard deviation of Gaussian noise.
        apply_to_contact_faces: If True, also apply noise to contact_faces coordinates.
    """
    if isinstance(contacts_or_std, (int, float)):
        return partial(_add_gaussian_noise_impl, std=contacts_or_std, apply_to_contact_faces=apply_to_contact_faces)
    return _add_gaussian_noise_impl(contacts_or_std, std, apply_to_contact_faces)


@builder.register("apply_random_rotation")
def apply_random_rotation(
    contacts: Sequence[SegContact], apply_to_contact_faces: bool = True
) -> Sequence[SegContact]:
    """Apply random 3D rotation around origin.

    Args:
        contacts: Sequence of SegContact objects to rotate.
        apply_to_contact_faces: If True, also rotate contact_faces coordinates.
    """
    result = []
    for contact in contacts:
        rot_matrix = _random_rotation_matrix()

        if apply_to_contact_faces:
            new_contact_faces = contact.contact_faces.copy()
            new_contact_faces[:, :3] = contact.contact_faces[:, :3] @ rot_matrix.T
        else:
            new_contact_faces = contact.contact_faces

        new_local_pointclouds = None
        if contact.local_pointclouds is not None:
            new_local_pointclouds = {}
            for config_tuple, pc_data in contact.local_pointclouds.items():
                new_local_pointclouds[config_tuple] = {
                    seg_id: (points @ rot_matrix.T).astype(np.float32)
                    for seg_id, points in pc_data.items()
                }

        result.append(
            SegContact(
                id=contact.id,
                seg_a=contact.seg_a,
                seg_b=contact.seg_b,
                com=contact.com,
                contact_faces=new_contact_faces,
                representative_points=contact.representative_points,
                local_pointclouds=new_local_pointclouds,
                merge_decisions=contact.merge_decisions,
                partner_metadata=contact.partner_metadata,
                contact_faces_original_nm=contact.contact_faces_original_nm,
            )
        )

    return result


@builder.register("apply_random_flip")
def apply_random_flip(
    contacts: Sequence[SegContact], apply_to_contact_faces: bool = True
) -> Sequence[SegContact]:
    """Randomly flip coordinates along x, y, z axes (50% probability each).

    Args:
        contacts: Sequence of SegContact objects to flip.
        apply_to_contact_faces: If True, also flip contact_faces coordinates.
    """
    result = []
    for contact in contacts:
        # Generate random flip signs for each axis
        flip_signs = np.where(np.random.random(3) < 0.5, -1.0, 1.0).astype(np.float32)

        if apply_to_contact_faces:
            new_contact_faces = contact.contact_faces.copy()
            new_contact_faces[:, :3] = contact.contact_faces[:, :3] * flip_signs
        else:
            new_contact_faces = contact.contact_faces

        new_local_pointclouds = None
        if contact.local_pointclouds is not None:
            new_local_pointclouds = {}
            for config_tuple, pc_data in contact.local_pointclouds.items():
                new_local_pointclouds[config_tuple] = {
                    seg_id: (points * flip_signs).astype(np.float32)
                    for seg_id, points in pc_data.items()
                }

        result.append(
            SegContact(
                id=contact.id,
                seg_a=contact.seg_a,
                seg_b=contact.seg_b,
                com=contact.com,
                contact_faces=new_contact_faces,
                representative_points=contact.representative_points,
                local_pointclouds=new_local_pointclouds,
                merge_decisions=contact.merge_decisions,
                partner_metadata=contact.partner_metadata,
                contact_faces_original_nm=contact.contact_faces_original_nm,
            )
        )

    return result


@builder.register("randomize_segment_identity")
def randomize_segment_identity(contacts: Sequence[SegContact]) -> Sequence[SegContact]:
    """Randomly swap seg_a and seg_b for each contact (50% probability).

    This effectively randomizes which segment gets label -1 vs +1 when the
    dataset assigns symmetric identity labels.
    """
    result = []
    for contact in contacts:
        if np.random.random() < 0.5:
            # Swap seg_a and seg_b
            new_local_pointclouds = None
            if contact.local_pointclouds is not None:
                new_local_pointclouds = {}
                for config_tuple, pc_data in contact.local_pointclouds.items():
                    # Swap the segment keys
                    new_local_pointclouds[config_tuple] = {
                        contact.seg_b: pc_data.get(contact.seg_a),
                        contact.seg_a: pc_data.get(contact.seg_b),
                    }

            new_partner_metadata = None
            if contact.partner_metadata is not None:
                new_partner_metadata = {
                    contact.seg_b: contact.partner_metadata.get(contact.seg_a),
                    contact.seg_a: contact.partner_metadata.get(contact.seg_b),
                }

            # Swap representative_points keys to match new seg_a/seg_b
            new_representative_points = {
                contact.seg_b: contact.representative_points.get(contact.seg_a),
                contact.seg_a: contact.representative_points.get(contact.seg_b),
            }

            result.append(
                SegContact(
                    id=contact.id,
                    seg_a=contact.seg_b,
                    seg_b=contact.seg_a,
                    com=contact.com,
                    contact_faces=contact.contact_faces,
                    representative_points=new_representative_points,
                    local_pointclouds=new_local_pointclouds,
                    merge_decisions=contact.merge_decisions,
                    partner_metadata=new_partner_metadata,
                    contact_faces_original_nm=contact.contact_faces_original_nm,
                )
            )
        else:
            # Keep as-is
            result.append(contact)

    return result

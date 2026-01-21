from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
from scipy.spatial.transform import Rotation

from zetta_utils import builder

from .contact import SegContact


def _random_rotation_matrix() -> np.ndarray:
    """Generate a random 3D rotation matrix."""
    return Rotation.random().as_matrix().astype(np.float32)


@builder.register("normalize_pointclouds")
def normalize_pointclouds(
    contacts: Sequence[SegContact], normalization_radius_nm: float = 8000.0
) -> Sequence[SegContact]:
    """Normalize all pointclouds and contact_faces to [-1, 1] centered on COM.

    Args:
        contacts: Sequence of SegContact objects to normalize.
        normalization_radius_nm: Radius in nm to use for normalizing contact_faces.
            For local_pointclouds, uses each config's radius_nm.
    """
    result = []
    for contact in contacts:
        com = np.array(contact.com, dtype=np.float32)

        # Normalize contact_faces: center on COM and scale by normalization_radius_nm
        new_contact_faces = contact.contact_faces.copy()
        new_contact_faces[:, :3] = (contact.contact_faces[:, :3] - com) / normalization_radius_nm

        # Normalize local_pointclouds if present
        new_local_pointclouds = None
        if contact.local_pointclouds is not None:
            new_local_pointclouds = {}
            for config_tuple, pc_data in contact.local_pointclouds.items():
                radius_nm = float(config_tuple[0])
                new_local_pointclouds[config_tuple] = {
                    seg_id: (points - com) / radius_nm for seg_id, points in pc_data.items()
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
            )
        )

    return result


def _add_gaussian_noise_impl(contacts: Sequence[SegContact], std: float) -> Sequence[SegContact]:
    """Implementation for adding Gaussian noise."""
    result = []
    for contact in contacts:
        if contact.local_pointclouds is None:
            result.append(contact)
            continue

        new_local_pointclouds = {}
        for config_tuple, pc_data in contact.local_pointclouds.items():
            new_local_pointclouds[config_tuple] = {
                seg_id: points + np.random.normal(0, std, points.shape).astype(np.float32)
                for seg_id, points in pc_data.items()
            }

        result.append(
            SegContact(
                id=contact.id,
                seg_a=contact.seg_a,
                seg_b=contact.seg_b,
                com=contact.com,
                contact_faces=contact.contact_faces,
                representative_points=contact.representative_points,
                local_pointclouds=new_local_pointclouds,
                merge_decisions=contact.merge_decisions,
                partner_metadata=contact.partner_metadata,
            )
        )

    return result


@builder.register("add_gaussian_noise")
def add_gaussian_noise(
    contacts_or_std: Sequence[SegContact] | float = 0.001,
    std: float = 0.001,
) -> Sequence[SegContact] | Callable:
    """Add Gaussian noise to pointcloud coordinates.

    Can be called two ways:
    - add_gaussian_noise(contacts) - uses default std=0.001
    - add_gaussian_noise(std=0.01) - returns a partial for use as processor
    """
    if isinstance(contacts_or_std, (int, float)):
        return partial(_add_gaussian_noise_impl, std=contacts_or_std)
    return _add_gaussian_noise_impl(contacts_or_std, std)


@builder.register("apply_random_rotation")
def apply_random_rotation(contacts: Sequence[SegContact]) -> Sequence[SegContact]:
    """Apply random 3D rotation around origin."""
    result = []
    for contact in contacts:
        rot_matrix = _random_rotation_matrix()

        new_contact_faces = contact.contact_faces.copy()
        new_contact_faces[:, :3] = contact.contact_faces[:, :3] @ rot_matrix.T

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
            )
        )

    return result

import numpy as np
import pytest
import torch

from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric.seg_contact import SegContact
from zetta_utils.layer.volumetric.seg_contact.tensor_utils import (
    SEG_A_LABEL,
    SEG_B_LABEL,
    contact_faces_to_tensor,
    contact_to_tensor,
    contacts_to_tensor,
    pointcloud_to_labeled_tensor,
)

# --- pointcloud_to_labeled_tensor ---


def test_pointcloud_to_labeled_tensor_basic():
    seg_a_pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    seg_b_pts = np.array([[7, 8, 9]], dtype=np.float32)

    a_tensor, b_tensor = pointcloud_to_labeled_tensor(seg_a_pts, seg_b_pts)

    assert a_tensor.shape == (2, 4)
    assert b_tensor.shape == (1, 4)
    # xyz preserved
    torch.testing.assert_close(a_tensor[:, :3], torch.tensor(seg_a_pts))
    torch.testing.assert_close(b_tensor[:, :3], torch.tensor(seg_b_pts))
    # labels
    assert (a_tensor[:, 3] == SEG_A_LABEL).all()
    assert (b_tensor[:, 3] == SEG_B_LABEL).all()


def test_pointcloud_to_labeled_tensor_with_affinity_channel():
    seg_a_pts = np.array([[1, 2, 3]], dtype=np.float32)
    seg_b_pts = np.array([[4, 5, 6]], dtype=np.float32)

    a_tensor, b_tensor = pointcloud_to_labeled_tensor(seg_a_pts, seg_b_pts, affinity_channel=True)

    assert a_tensor.shape == (1, 5)
    assert b_tensor.shape == (1, 5)
    # 5th channel is 0 for segment points
    assert a_tensor[0, 4] == 0.0
    assert b_tensor[0, 4] == 0.0


def test_pointcloud_to_labeled_tensor_empty():
    seg_a_pts = np.zeros((0, 3), dtype=np.float32)
    seg_b_pts = np.array([[1, 2, 3]], dtype=np.float32)

    a_tensor, b_tensor = pointcloud_to_labeled_tensor(seg_a_pts, seg_b_pts)

    assert a_tensor.shape == (0, 4)
    assert b_tensor.shape == (1, 4)


# --- contact_faces_to_tensor ---


def test_contact_faces_to_tensor_default_label():
    faces = np.array([[1, 2, 3, 0.8], [4, 5, 6, 0.6]], dtype=np.float32)

    result = contact_faces_to_tensor(faces, contact_label=0.0)

    assert result is not None
    assert result.shape == (2, 4)
    # xyz preserved
    torch.testing.assert_close(result[:, :3], torch.tensor(faces[:, :3]))
    # 4th channel replaced with contact_label
    assert (result[:, 3] == 0.0).all()


def test_contact_faces_to_tensor_label_none_keeps_affinity():
    faces = np.array([[1, 2, 3, 0.8], [4, 5, 6, 0.6]], dtype=np.float32)

    result = contact_faces_to_tensor(faces, contact_label=None)

    assert result is not None
    # 4th channel keeps original affinity
    torch.testing.assert_close(result[:, 3], torch.tensor([0.8, 0.6]))


def test_contact_faces_to_tensor_per_point_affinity():
    faces = np.array([[1, 2, 3, 0.8], [4, 5, 6, 0.6]], dtype=np.float32)

    result = contact_faces_to_tensor(faces, contact_label=0.0, affinity_channel_mode="per_point")

    assert result is not None
    assert result.shape == (2, 5)
    # 5th channel has per-point affinity
    torch.testing.assert_close(result[:, 4], torch.tensor([0.8, 0.6]))


def test_contact_faces_to_tensor_mean_affinity():
    faces = np.array([[1, 2, 3, 0.8], [4, 5, 6, 0.6]], dtype=np.float32)

    result = contact_faces_to_tensor(faces, contact_label=0.0, affinity_channel_mode="mean")

    assert result is not None
    assert result.shape == (2, 5)
    # 5th channel has mean affinity broadcast
    expected_mean = (0.8 + 0.6) / 2
    torch.testing.assert_close(result[:, 4], torch.tensor([expected_mean, expected_mean]))


def test_contact_faces_to_tensor_empty_returns_none():
    faces = np.zeros((0, 4), dtype=np.float32)
    result = contact_faces_to_tensor(faces)
    assert result is None


def test_contact_faces_to_tensor_unknown_mode_raises():
    faces = np.array([[1, 2, 3, 0.5]], dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown affinity_channel_mode"):
        contact_faces_to_tensor(faces, affinity_channel_mode="invalid")


# --- contact_to_tensor ---


def _make_contact(
    n_a=5,
    n_b=3,
    config_key=(500, 64),
    include_faces=True,
    n_faces=2,
):
    seg_a_pts = np.random.randn(n_a, 3).astype(np.float32)
    seg_b_pts = np.random.randn(n_b, 3).astype(np.float32)
    faces = (
        np.random.randn(n_faces, 4).astype(np.float32)
        if include_faces
        else np.zeros((0, 4), dtype=np.float32)
    )
    return SegContact(
        id=1,
        seg_a=100,
        seg_b=200,
        com=Vec3D(0.0, 0.0, 0.0),
        contact_faces=faces,
        local_pointclouds={config_key: {100: seg_a_pts, 200: seg_b_pts}},
    )


def test_contact_to_tensor_segments_only():
    contact = _make_contact(n_a=5, n_b=3)
    result = contact_to_tensor(contact, config_key=(500, 64), include_contact_faces=False)

    assert result is not None
    assert result.shape == (8, 4)  # 5 + 3 points, 4 channels


def test_contact_to_tensor_with_contact_faces():
    contact = _make_contact(n_a=5, n_b=3, n_faces=4)
    result = contact_to_tensor(contact, config_key=(500, 64), include_contact_faces=True)

    assert result is not None
    assert result.shape == (12, 4)  # 5 + 3 + 4 points


def test_contact_to_tensor_with_affinity_channel():
    contact = _make_contact(n_a=5, n_b=3, n_faces=2)
    result = contact_to_tensor(
        contact,
        config_key=(500, 64),
        include_contact_faces=True,
        affinity_channel_mode="per_point",
    )

    assert result is not None
    assert result.shape == (10, 5)  # 5 + 3 + 2 points, 5 channels


def test_contact_to_tensor_missing_config_returns_none():
    contact = _make_contact(config_key=(500, 64))
    result = contact_to_tensor(contact, config_key=(999, 99))
    assert result is None


def test_contact_to_tensor_no_pointclouds_returns_none():
    contact = SegContact(
        id=1,
        seg_a=100,
        seg_b=200,
        com=Vec3D(0.0, 0.0, 0.0),
        contact_faces=np.zeros((0, 4), dtype=np.float32),
        local_pointclouds=None,
    )
    result = contact_to_tensor(contact, config_key=(500, 64))
    assert result is None


# --- contacts_to_tensor ---


def test_contacts_to_tensor_batch():
    contacts = [_make_contact(n_a=5, n_b=5) for _ in range(3)]
    tensor, valid_indices = contacts_to_tensor(contacts, config_key=(500, 64))

    assert tensor.shape[0] == 3  # batch size
    assert tensor.shape[1] == 4  # channels (xyz + label)
    assert tensor.shape[2] == 10  # 5 + 5 points
    assert valid_indices == [0, 1, 2]


def test_contacts_to_tensor_skips_invalid():
    good = _make_contact(n_a=5, n_b=5, config_key=(500, 64))
    bad = SegContact(
        id=2,
        seg_a=300,
        seg_b=400,
        com=Vec3D(0.0, 0.0, 0.0),
        contact_faces=np.zeros((0, 4), dtype=np.float32),
        local_pointclouds=None,
    )

    tensor, valid_indices = contacts_to_tensor([bad, good, bad], config_key=(500, 64))

    assert tensor.shape[0] == 1
    assert valid_indices == [1]


def test_contacts_to_tensor_empty():
    tensor, valid_indices = contacts_to_tensor([], config_key=(500, 64))
    assert tensor.shape == (0, 4, 0)
    assert valid_indices == []


def test_contacts_to_tensor_auto_config_key():
    contacts = [_make_contact(n_a=4, n_b=4, config_key=(300, 32))]
    tensor, valid_indices = contacts_to_tensor(contacts, config_key=None)

    assert tensor.shape[0] == 1
    assert tensor.shape[2] == 8
    assert valid_indices == [0]


def test_contacts_to_tensor_transposed():
    """Verify output is [B, C, N] not [B, N, C]."""
    contacts = [_make_contact(n_a=3, n_b=3)]
    tensor, _ = contacts_to_tensor(contacts, config_key=(500, 64))

    # C=4, N=6
    assert tensor.shape == (1, 4, 6)

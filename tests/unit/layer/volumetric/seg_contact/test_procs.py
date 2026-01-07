import numpy as np

from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric.seg_contact import (
    SegContact,
    add_gaussian_noise,
    apply_random_rotation,
    normalize_pointclouds,
)


def test_normalize_pointclouds_basic():
    """Test normalization centers on COM and scales by radius from tuple key."""
    com = Vec3D(100.0, 200.0, 300.0)
    config = (500, 64)  # (radius_nm, n_points)

    # Create pointcloud at known positions
    seg_a_pts = np.array([[100, 200, 300], [600, 200, 300]], dtype=np.float32)
    seg_b_pts = np.array([[100, 700, 300], [100, 200, 800]], dtype=np.float32)

    contact = SegContact(
        id=1,
        seg_a=100,
        seg_b=200,
        com=com,
        contact_faces=np.array([[1, 2, 3, 0.5]], dtype=np.float32),
        local_pointclouds={config: {100: seg_a_pts, 200: seg_b_pts}},
    )

    result = normalize_pointclouds([contact])

    assert len(result) == 1
    normalized = result[0]
    assert normalized.local_pointclouds is not None

    # Check normalization: (point - com) / radius
    # seg_a point 0: (100-100, 200-200, 300-300) / 500 = (0, 0, 0)
    # seg_a point 1: (600-100, 200-200, 300-300) / 500 = (1, 0, 0)
    expected_a = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    # seg_b point 0: (100-100, 700-200, 300-300) / 500 = (0, 1, 0)
    # seg_b point 1: (100-100, 200-200, 800-300) / 500 = (0, 0, 1)
    expected_b = np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32)

    np.testing.assert_array_almost_equal(normalized.local_pointclouds[config][100], expected_a)
    np.testing.assert_array_almost_equal(normalized.local_pointclouds[config][200], expected_b)


def test_normalize_pointclouds_no_pointclouds():
    """Test proc normalizes contact_faces even without local_pointclouds."""
    com = Vec3D(100.0, 100.0, 100.0)
    contact = SegContact(
        id=1,
        seg_a=100,
        seg_b=200,
        com=com,
        contact_faces=np.array([[100, 200, 300, 0.5]], dtype=np.float32),
        local_pointclouds=None,
    )

    # Use default normalization_radius_nm=8000
    result = normalize_pointclouds([contact])

    assert len(result) == 1
    normalized = result[0]
    assert normalized.local_pointclouds is None
    # contact_faces should be normalized: (xyz - com) / normalization_radius_nm
    # (100-100, 200-100, 300-100) / 8000 = (0, 0.0125, 0.025)
    expected_faces = np.array([[0, 0.0125, 0.025, 0.5]], dtype=np.float32)
    np.testing.assert_array_almost_equal(normalized.contact_faces, expected_faces)


def test_normalize_pointclouds_multiple_configs():
    """Test proc normalizes all pointcloud configs using each config's radius."""
    com = Vec3D(0.0, 0.0, 0.0)
    config1 = (100, 32)  # radius=100nm
    config2 = (500, 64)  # radius=500nm

    # Point at (100, 0, 0) should normalize to (1, 0, 0) for config1, (0.2, 0, 0) for config2
    pts = np.array([[100, 0, 0]], dtype=np.float32)

    contact = SegContact(
        id=1,
        seg_a=100,
        seg_b=200,
        com=com,
        contact_faces=np.array([[1, 2, 3, 0.5]], dtype=np.float32),
        local_pointclouds={
            config1: {100: pts.copy(), 200: pts.copy()},
            config2: {100: pts.copy(), 200: pts.copy()},
        },
    )

    result = normalize_pointclouds([contact])

    normalized = result[0]
    assert normalized.local_pointclouds is not None
    # config1: 100/100 = 1.0
    np.testing.assert_array_almost_equal(normalized.local_pointclouds[config1][100], [[1.0, 0, 0]])
    # config2: 100/500 = 0.2
    np.testing.assert_array_almost_equal(normalized.local_pointclouds[config2][100], [[0.2, 0, 0]])


def test_normalize_pointclouds_preserves_other_fields():
    """Test normalization preserves metadata fields and normalizes contact_faces."""
    config = (500, 64)
    com = Vec3D(100.0, 200.0, 300.0)
    contact = SegContact(
        id=42,
        seg_a=100,
        seg_b=200,
        com=com,
        contact_faces=np.array([[100, 200, 300, 0.5], [8100, 200, 300, 0.8]], dtype=np.float32),
        local_pointclouds={config: {100: np.zeros((10, 3)), 200: np.ones((10, 3))}},
        merge_decisions={"human": True},
        partner_metadata={100: "axon", 200: "dendrite"},
    )

    result = normalize_pointclouds([contact])

    normalized = result[0]
    assert normalized.id == 42
    assert normalized.seg_a == 100
    assert normalized.seg_b == 200
    assert normalized.com == Vec3D(100, 200, 300)
    assert normalized.merge_decisions == {"human": True}
    assert normalized.partner_metadata == {100: "axon", 200: "dendrite"}
    # contact_faces should be normalized: (xyz - com) / normalization_radius_nm (default 8000)
    # Point 0: (100-100, 200-200, 300-300) / 8000 = (0, 0, 0)
    # Point 1: (8100-100, 200-200, 300-300) / 8000 = (1, 0, 0)
    expected_faces = np.array([[0, 0, 0, 0.5], [1, 0, 0, 0.8]], dtype=np.float32)
    np.testing.assert_array_almost_equal(normalized.contact_faces, expected_faces)


def test_gaussian_noise_adds_noise():
    """Test Gaussian noise is added to pointcloud coordinates."""
    config = (500, 64)
    pts = np.zeros((100, 3), dtype=np.float32)

    contact = SegContact(
        id=1,
        seg_a=100,
        seg_b=200,
        com=Vec3D(0.0, 0.0, 0.0),
        contact_faces=np.array([[1, 2, 3, 0.5]], dtype=np.float32),
        local_pointclouds={config: {100: pts.copy(), 200: pts.copy()}},
    )

    proc = add_gaussian_noise(std=0.01)
    result = proc([contact])  # type: ignore[operator]

    noisy = result[0]
    assert noisy.local_pointclouds is not None
    # Noise should have been added (not exactly zero anymore)
    assert not np.allclose(noisy.local_pointclouds[config][100], 0)
    assert not np.allclose(noisy.local_pointclouds[config][200], 0)
    # But should be small (within ~3 std)
    assert np.abs(noisy.local_pointclouds[config][100]).max() < 0.1
    assert np.abs(noisy.local_pointclouds[config][200]).max() < 0.1


def test_gaussian_noise_no_pointclouds():
    """Test Gaussian noise proc passes through contacts without pointclouds."""
    contact = SegContact(
        id=1,
        seg_a=100,
        seg_b=200,
        com=Vec3D(0.0, 0.0, 0.0),
        contact_faces=np.array([[1, 2, 3, 0.5]], dtype=np.float32),
        local_pointclouds=None,
    )

    proc = add_gaussian_noise()
    result = proc([contact])  # type: ignore[operator]

    assert len(result) == 1
    assert result[0] is contact


def test_gaussian_noise_partial_with_std():
    """Test add_gaussian_noise returns partial when called with std parameter."""
    config = (500, 64)
    pts = np.zeros((100, 3), dtype=np.float32)

    contact = SegContact(
        id=1,
        seg_a=100,
        seg_b=200,
        com=Vec3D(0.0, 0.0, 0.0),
        contact_faces=np.array([[1, 2, 3, 0.5]], dtype=np.float32),
        local_pointclouds={config: {100: pts.copy(), 200: pts.copy()}},
    )

    # Call with std parameter to get a partial
    proc = add_gaussian_noise(0.02)
    result = proc([contact])  # type: ignore[operator]

    noisy = result[0]
    assert noisy.local_pointclouds is not None
    # Noise should have been added
    assert not np.allclose(noisy.local_pointclouds[config][100], 0)


def test_gaussian_noise_direct_call_with_contacts():
    """Test add_gaussian_noise called directly with contacts."""
    config = (500, 64)
    pts = np.zeros((100, 3), dtype=np.float32)

    contact = SegContact(
        id=1,
        seg_a=100,
        seg_b=200,
        com=Vec3D(0.0, 0.0, 0.0),
        contact_faces=np.array([[1, 2, 3, 0.5]], dtype=np.float32),
        local_pointclouds={config: {100: pts.copy(), 200: pts.copy()}},
    )

    # Call directly with contacts (not via partial)
    result = add_gaussian_noise([contact])

    noisy = result[0]  # type: ignore[index]
    assert noisy.local_pointclouds is not None
    # Noise should have been added with default std
    assert not np.allclose(noisy.local_pointclouds[config][100], 0)


def test_random_rotation_rotates_pointclouds_and_faces():
    """Test rotation is applied to both pointclouds and contact_faces."""
    config = (500, 64)
    pts_a = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    pts_b = np.array([[0, 0, 1]], dtype=np.float32)
    faces = np.array([[1, 0, 0, 0.5], [0, 1, 0, 0.8]], dtype=np.float32)

    contact = SegContact(
        id=1,
        seg_a=100,
        seg_b=200,
        com=Vec3D(0.0, 0.0, 0.0),
        contact_faces=faces.copy(),
        local_pointclouds={config: {100: pts_a.copy(), 200: pts_b.copy()}},
    )

    result = apply_random_rotation([contact])

    rotated = result[0]
    assert rotated.local_pointclouds is not None

    # Points should be rotated (different from original)
    assert not np.allclose(rotated.local_pointclouds[config][100], pts_a)
    # But norms should be preserved (rotation is length-preserving)
    np.testing.assert_array_almost_equal(
        np.linalg.norm(rotated.local_pointclouds[config][100], axis=1),
        np.linalg.norm(pts_a, axis=1),
    )
    np.testing.assert_array_almost_equal(
        np.linalg.norm(rotated.local_pointclouds[config][200], axis=1),
        np.linalg.norm(pts_b, axis=1),
    )

    # Contact faces xyz should be rotated
    assert not np.allclose(rotated.contact_faces[:, :3], faces[:, :3])
    # But norms preserved
    np.testing.assert_array_almost_equal(
        np.linalg.norm(rotated.contact_faces[:, :3], axis=1),
        np.linalg.norm(faces[:, :3], axis=1),
    )
    # 4th column (affinity) should be unchanged
    np.testing.assert_array_equal(rotated.contact_faces[:, 3], faces[:, 3])


def test_random_rotation_no_pointclouds():
    """Test rotation proc still rotates contact_faces when no pointclouds."""
    faces = np.array([[1, 0, 0, 0.5]], dtype=np.float32)

    contact = SegContact(
        id=1,
        seg_a=100,
        seg_b=200,
        com=Vec3D(0.0, 0.0, 0.0),
        contact_faces=faces.copy(),
        local_pointclouds=None,
    )

    result = apply_random_rotation([contact])

    rotated = result[0]
    # Should still rotate faces
    np.testing.assert_array_almost_equal(
        np.linalg.norm(rotated.contact_faces[:, :3], axis=1),
        np.linalg.norm(faces[:, :3], axis=1),
    )
    assert rotated.local_pointclouds is None


def test_random_rotation_preserves_other_fields():
    """Test rotation preserves all other contact fields."""
    config = (500, 64)
    contact = SegContact(
        id=42,
        seg_a=100,
        seg_b=200,
        com=Vec3D(100.0, 200.0, 300.0),
        contact_faces=np.array([[1, 2, 3, 0.5]], dtype=np.float32),
        local_pointclouds={config: {100: np.zeros((10, 3)), 200: np.ones((10, 3))}},
        merge_decisions={"human": True},
        partner_metadata={100: "axon", 200: "dendrite"},
    )

    result = apply_random_rotation([contact])

    rotated = result[0]
    assert rotated.id == 42
    assert rotated.seg_a == 100
    assert rotated.seg_b == 200
    assert rotated.com == Vec3D(100, 200, 300)
    assert rotated.merge_decisions == {"human": True}
    assert rotated.partner_metadata == {100: "axon", 200: "dendrite"}

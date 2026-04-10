import numpy as np
import pytest

from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric.seg_contact import (
    SegContact,
    add_gaussian_noise,
    apply_random_rotation,
    normalize_pointclouds,
)
from zetta_utils.layer.volumetric.seg_contact.procs import (
    apply_random_flip,
    deduplicate_pointclouds,
    randomize_segment_identity,
    resample_combined_pointcloud,
    resample_pointclouds,
    resample_points,
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
        representative_points={100: Vec3D(90.0, 190.0, 290.0), 200: Vec3D(110.0, 210.0, 310.0)},
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
        representative_points={100: Vec3D(90.0, 90.0, 90.0), 200: Vec3D(110.0, 110.0, 110.0)},
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
        representative_points={100: Vec3D(-10.0, -10.0, -10.0), 200: Vec3D(10.0, 10.0, 10.0)},
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
        representative_points={100: Vec3D(90.0, 190.0, 290.0), 200: Vec3D(110.0, 210.0, 310.0)},
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
        representative_points={100: Vec3D(-10.0, -10.0, -10.0), 200: Vec3D(10.0, 10.0, 10.0)},
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
        representative_points={100: Vec3D(-10.0, -10.0, -10.0), 200: Vec3D(10.0, 10.0, 10.0)},
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
        representative_points={100: Vec3D(-10.0, -10.0, -10.0), 200: Vec3D(10.0, 10.0, 10.0)},
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
        representative_points={100: Vec3D(-10.0, -10.0, -10.0), 200: Vec3D(10.0, 10.0, 10.0)},
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
        representative_points={100: Vec3D(-10.0, -10.0, -10.0), 200: Vec3D(10.0, 10.0, 10.0)},
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
        representative_points={100: Vec3D(-10.0, -10.0, -10.0), 200: Vec3D(10.0, 10.0, 10.0)},
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
        representative_points={100: Vec3D(90.0, 190.0, 290.0), 200: Vec3D(110.0, 210.0, 310.0)},
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


# --- Helper for new tests ---


def _make_contact(n_a=10, n_b=10, n_faces=5, config_key=(500, 64)):
    seg_a_pts = np.random.randn(n_a, 3).astype(np.float32)
    seg_b_pts = np.random.randn(n_b, 3).astype(np.float32)
    faces = np.random.randn(n_faces, 4).astype(np.float32)
    faces[:, 3] = np.abs(faces[:, 3])  # positive affinity
    return SegContact(
        id=1,
        seg_a=100,
        seg_b=200,
        com=Vec3D(0.0, 0.0, 0.0),
        contact_faces=faces,
        representative_points={100: Vec3D(-1.0, 0.0, 0.0), 200: Vec3D(1.0, 0.0, 0.0)},
        local_pointclouds={config_key: {100: seg_a_pts, 200: seg_b_pts}},
        partner_metadata={100: "axon", 200: "dendrite"},
    )


# --- resample_points ---


def test_resample_points_uniform_downsample():
    pts = np.random.randn(100, 3).astype(np.float32)
    result = resample_points(pts, 50, weighting="uniform")
    assert result.shape == (50, 3)


def test_resample_points_uniform_upsample():
    pts = np.random.randn(10, 3).astype(np.float32)
    result = resample_points(pts, 50, weighting="uniform")
    assert result.shape == (50, 3)


def test_resample_points_uniform_noop():
    pts = np.random.randn(10, 3).astype(np.float32)
    result = resample_points(pts, 10, weighting="uniform")
    assert result is pts


def test_resample_points_inverse_r_favors_center():
    pts = np.array([[0.1, 0, 0], [10, 0, 0], [100, 0, 0]], dtype=np.float32)
    np.random.seed(42)
    result = resample_points(pts, 1000, weighting="inverse_r")
    assert result.shape == (1000, 3)
    close_count = np.sum(np.abs(result[:, 0] - 0.1) < 0.01)
    far_count = np.sum(np.abs(result[:, 0] - 100) < 0.01)
    assert close_count > far_count


def test_resample_points_inverse_r2():
    pts = np.array([[0.1, 0, 0], [10, 0, 0]], dtype=np.float32)
    np.random.seed(42)
    result = resample_points(pts, 500, weighting="inverse_r2")
    assert result.shape == (500, 3)


def test_resample_points_unknown_weighting_raises():
    pts = np.random.randn(10, 3).astype(np.float32)
    with pytest.raises(ValueError, match="Unknown weighting"):
        resample_points(pts, 5, weighting="bad")


def test_resample_points_empty_input():
    pts = np.zeros((0, 3), dtype=np.float32)
    result = resample_points(pts, 5, weighting="uniform")
    assert result.shape == (5, 3)
    assert np.allclose(result, 0)


def test_resample_points_zero_target():
    pts = np.random.randn(10, 3).astype(np.float32)
    result = resample_points(pts, 0, weighting="uniform")
    assert result.shape == (0, 3)


def test_resample_points_return_indices():
    pts = np.random.randn(20, 3).astype(np.float32)
    result, indices = resample_points(pts, 10, weighting="uniform", return_indices=True)
    assert result.shape == (10, 3)
    assert indices.shape == (10,)
    np.testing.assert_array_equal(result, pts[indices])


def test_resample_points_return_indices_noop():
    pts = np.random.randn(10, 3).astype(np.float32)
    result, indices = resample_points(pts, 10, weighting="uniform", return_indices=True)
    assert result is pts
    np.testing.assert_array_equal(indices, np.arange(10))


def test_resample_points_custom_center():
    pts = np.array([[10, 0, 0], [0, 10, 0]], dtype=np.float32)
    center = np.array([10, 0, 0], dtype=np.float32)
    np.random.seed(42)
    result = resample_points(pts, 500, weighting="inverse_r", center=center)
    close_count = np.sum(np.abs(result[:, 0] - 10) < 0.01)
    assert close_count > 250


def test_resample_points_preserves_extra_dimensions():
    pts = np.random.randn(20, 5).astype(np.float32)
    result = resample_points(pts, 10, weighting="uniform")
    assert result.shape == (10, 5)


# --- resample_pointclouds ---


def test_resample_pointclouds_segment_target():
    contact = _make_contact(n_a=50, n_b=30)
    result = resample_pointclouds([contact], segment_target=20)
    pc = result[0].local_pointclouds[(500, 64)]
    assert pc[100].shape[0] == 20
    assert pc[200].shape[0] == 20


def test_resample_pointclouds_contact_face_target():
    contact = _make_contact(n_faces=20)
    result = resample_pointclouds([contact], contact_face_target=5)
    assert result[0].contact_faces.shape[0] == 5


def test_resample_pointclouds_both():
    contact = _make_contact(n_a=50, n_b=30, n_faces=20)
    result = resample_pointclouds([contact], segment_target=15, contact_face_target=8)
    pc = result[0].local_pointclouds[(500, 64)]
    assert pc[100].shape[0] == 15
    assert pc[200].shape[0] == 15
    assert result[0].contact_faces.shape[0] == 8


def test_resample_pointclouds_preserves_metadata():
    contact = _make_contact()
    contact.merge_decisions = {"human": True}
    result = resample_pointclouds([contact], segment_target=5)
    assert result[0].merge_decisions == {"human": True}
    assert result[0].id == 1
    assert result[0].seg_a == 100
    assert result[0].seg_b == 200


# --- resample_combined_pointcloud ---


def test_resample_combined_total_matches_target():
    contact = _make_contact(n_a=100, n_b=100, n_faces=0)
    result = resample_combined_pointcloud([contact], total_target=200, include_contact_faces=False)
    pc = result[0].local_pointclouds[(500, 64)]
    total = pc[100].shape[0] + pc[200].shape[0] + result[0].contact_faces.shape[0]
    assert total == 200


def test_resample_combined_cf_fraction_capped():
    contact = _make_contact(n_a=100, n_b=100, n_faces=50)
    result = resample_combined_pointcloud(
        [contact],
        total_target=200,
        include_contact_faces=True,
        max_contact_face_fraction=0.1,
    )
    pc = result[0].local_pointclouds[(500, 64)]
    total = pc[100].shape[0] + pc[200].shape[0] + result[0].contact_faces.shape[0]
    assert total == 200
    assert result[0].contact_faces.shape[0] <= 20


def test_resample_combined_cf_capped_by_actual_count():
    contact = _make_contact(n_a=100, n_b=100, n_faces=3)
    result = resample_combined_pointcloud(
        [contact],
        total_target=200,
        include_contact_faces=True,
        max_contact_face_fraction=0.5,
    )
    assert result[0].contact_faces.shape[0] == 3


def test_resample_combined_independent_splits_evenly():
    contact = _make_contact(n_a=100, n_b=100, n_faces=0)
    result = resample_combined_pointcloud(
        [contact],
        total_target=100,
        include_contact_faces=False,
        joint_segment_sampling=False,
    )
    pc = result[0].local_pointclouds[(500, 64)]
    assert pc[100].shape[0] == 50
    assert pc[200].shape[0] == 50


def test_resample_combined_joint_sampling():
    contact = _make_contact(n_a=100, n_b=100, n_faces=0)
    result = resample_combined_pointcloud(
        [contact],
        total_target=100,
        include_contact_faces=False,
        joint_segment_sampling=True,
    )
    pc = result[0].local_pointclouds[(500, 64)]
    total_seg = pc[100].shape[0] + pc[200].shape[0]
    assert total_seg == 100


def test_resample_combined_balanced_weighting():
    contact = _make_contact(n_a=100, n_b=100, n_faces=0)
    result = resample_combined_pointcloud(
        [contact],
        total_target=100,
        include_contact_faces=False,
        weighting="balanced",
        joint_segment_sampling=True,
    )
    pc = result[0].local_pointclouds[(500, 64)]
    total_seg = pc[100].shape[0] + pc[200].shape[0]
    assert total_seg == 100


def test_resample_combined_skips_no_pointclouds():
    contact = SegContact(
        id=1,
        seg_a=100,
        seg_b=200,
        com=Vec3D(0.0, 0.0, 0.0),
        contact_faces=np.zeros((0, 4), dtype=np.float32),
        local_pointclouds=None,
    )
    result = resample_combined_pointcloud([contact], total_target=100)
    assert result[0].local_pointclouds is None


# --- deduplicate_pointclouds ---


def test_deduplicate_removes_duplicates():
    pts = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]], dtype=np.float32)
    contact = _make_contact()
    contact.local_pointclouds = {(500, 64): {100: pts, 200: pts.copy()}}
    result = deduplicate_pointclouds([contact])
    assert result[0].local_pointclouds[(500, 64)][100].shape[0] == 2


def test_deduplicate_no_duplicates():
    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    contact = _make_contact()
    contact.local_pointclouds = {(500, 64): {100: pts, 200: pts.copy()}}
    result = deduplicate_pointclouds([contact])
    assert result[0].local_pointclouds[(500, 64)][100].shape[0] == 2


def test_deduplicate_contact_faces():
    faces = np.array([[1, 2, 3, 0.5], [1, 2, 3, 0.5], [4, 5, 6, 0.8]], dtype=np.float32)
    contact = _make_contact()
    contact.contact_faces = faces
    result = deduplicate_pointclouds([contact], apply_to_contact_faces=True)
    assert result[0].contact_faces.shape[0] == 2


def test_deduplicate_no_pointclouds():
    contact = SegContact(
        id=1,
        seg_a=100,
        seg_b=200,
        com=Vec3D(0.0, 0.0, 0.0),
        contact_faces=np.zeros((0, 4), dtype=np.float32),
        local_pointclouds=None,
    )
    result = deduplicate_pointclouds([contact])
    assert result[0].local_pointclouds is None


# --- apply_random_flip ---


def test_random_flip_preserves_magnitudes():
    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    contact = _make_contact()
    contact.local_pointclouds = {(500, 64): {100: pts.copy(), 200: pts.copy()}}
    contact.contact_faces = np.array([[1, 2, 3, 0.5]], dtype=np.float32)

    result = apply_random_flip([contact])
    r = result[0]
    np.testing.assert_array_almost_equal(np.abs(r.local_pointclouds[(500, 64)][100]), np.abs(pts))


def test_random_flip_preserves_affinity():
    contact = _make_contact()
    original_aff = contact.contact_faces[:, 3].copy()
    result = apply_random_flip([contact])
    np.testing.assert_array_equal(result[0].contact_faces[:, 3], original_aff)


def test_random_flip_eventually_flips():
    """At least one flip should occur over multiple trials."""
    pts = np.array([[1, 2, 3]], dtype=np.float32)
    any_flipped = False
    for _ in range(20):
        c = _make_contact()
        c.local_pointclouds = {(500, 64): {100: pts.copy(), 200: pts.copy()}}
        result = apply_random_flip([c])
        if not np.allclose(result[0].local_pointclouds[(500, 64)][100], pts):
            any_flipped = True
            break
    assert any_flipped


def test_random_flip_no_pointclouds():
    contact = SegContact(
        id=1,
        seg_a=100,
        seg_b=200,
        com=Vec3D(0.0, 0.0, 0.0),
        contact_faces=np.array([[1, 2, 3, 0.5]], dtype=np.float32),
        local_pointclouds=None,
    )
    result = apply_random_flip([contact])
    assert result[0].local_pointclouds is None


# --- randomize_segment_identity ---


def test_randomize_segment_eventually_swaps():
    swapped = False
    for _ in range(20):
        contact = _make_contact()
        result = randomize_segment_identity([contact])
        if result[0].seg_a == 200 and result[0].seg_b == 100:
            swapped = True
            break
    assert swapped, "Expected at least one swap in 20 trials"


def test_randomize_segment_swap_preserves_pointcloud_data():
    pts_a = np.array([[1, 0, 0]], dtype=np.float32)
    pts_b = np.array([[0, 1, 0]], dtype=np.float32)

    for _ in range(50):
        contact = _make_contact()
        contact.local_pointclouds = {(500, 64): {100: pts_a.copy(), 200: pts_b.copy()}}
        result = randomize_segment_identity([contact])
        r = result[0]
        if r.seg_a == 200:
            pc = r.local_pointclouds[(500, 64)]
            np.testing.assert_array_equal(pc[200], pts_a)
            np.testing.assert_array_equal(pc[100], pts_b)
            return
    pytest.fail("No swap occurred in 50 trials")


def test_randomize_segment_swap_preserves_partner_metadata():
    for _ in range(50):
        contact = _make_contact()
        result = randomize_segment_identity([contact])
        r = result[0]
        if r.seg_a == 200:
            assert r.partner_metadata[200] == "axon"
            assert r.partner_metadata[100] == "dendrite"
            return
    pytest.fail("No swap occurred in 50 trials")


def test_randomize_segment_swap_preserves_representative_points():
    for _ in range(50):
        contact = _make_contact()
        result = randomize_segment_identity([contact])
        r = result[0]
        if r.seg_a == 200:
            assert r.representative_points[200] == Vec3D(-1.0, 0.0, 0.0)
            assert r.representative_points[100] == Vec3D(1.0, 0.0, 0.0)
            return
    pytest.fail("No swap occurred in 50 trials")


def test_randomize_segment_no_pointclouds():
    contact = SegContact(
        id=1,
        seg_a=100,
        seg_b=200,
        com=Vec3D(0.0, 0.0, 0.0),
        contact_faces=np.zeros((0, 4), dtype=np.float32),
        local_pointclouds=None,
    )
    result = randomize_segment_identity([contact])
    assert result[0].seg_a in (100, 200)


# --- normalize_pointclouds with use_pointcloud_radius ---


def test_normalize_use_pointcloud_radius():
    """Test normalize uses the largest config's radius when use_pointcloud_radius=True."""
    com = Vec3D(0.0, 0.0, 0.0)
    config_small = (500, 32)
    config_large = (2000, 64)

    pts = np.array([[2000, 0, 0]], dtype=np.float32)
    faces = np.array([[2000, 0, 0, 0.5]], dtype=np.float32)

    contact = SegContact(
        id=1,
        seg_a=100,
        seg_b=200,
        com=com,
        contact_faces=faces,
        representative_points={100: Vec3D(-10.0, 0.0, 0.0), 200: Vec3D(10.0, 0.0, 0.0)},
        local_pointclouds={
            config_small: {100: pts.copy(), 200: pts.copy()},
            config_large: {100: pts.copy(), 200: pts.copy()},
        },
    )

    result = normalize_pointclouds([contact], use_pointcloud_radius=True)
    normalized = result[0]

    # With use_pointcloud_radius=True, largest config is (2000, 64), radius=2000
    # Both configs should use radius=2000 for normalization
    # 2000 / 2000 = 1.0
    np.testing.assert_array_almost_equal(
        normalized.local_pointclouds[config_large][100], [[1.0, 0, 0]]
    )
    np.testing.assert_array_almost_equal(
        normalized.local_pointclouds[config_small][100], [[1.0, 0, 0]]
    )
    # Contact faces also use the largest config radius
    np.testing.assert_array_almost_equal(normalized.contact_faces[:, :3], [[1.0, 0, 0]])


def test_normalize_use_pointcloud_radius_preserves_original_faces():
    """Test normalize stores contact_faces_original_nm before normalization."""
    com = Vec3D(100.0, 0.0, 0.0)
    config = (500, 32)
    faces = np.array([[200, 0, 0, 0.8]], dtype=np.float32)

    contact = SegContact(
        id=1,
        seg_a=100,
        seg_b=200,
        com=com,
        contact_faces=faces.copy(),
        representative_points={100: Vec3D(90.0, 0.0, 0.0), 200: Vec3D(110.0, 0.0, 0.0)},
        local_pointclouds={
            config: {
                100: np.zeros((5, 3), dtype=np.float32),
                200: np.zeros((5, 3), dtype=np.float32),
            }
        },
    )

    result = normalize_pointclouds([contact], use_pointcloud_radius=True)
    normalized = result[0]

    # Original faces should be preserved
    np.testing.assert_array_almost_equal(normalized.contact_faces_original_nm, faces)

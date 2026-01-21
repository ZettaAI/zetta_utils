import numpy as np

from zetta_utils.geometry import Vec3D
from zetta_utils.mazepa_layer_processing.common.seg_contact_op import (
    SegContactOp,
    _blackout_segments,
    _build_seg_to_ref,
    _build_voxel_spatial_hash,
    _compute_affinity_weighted_com,
    _compute_contact_connected_components,
    _compute_contact_counts,
    _compute_overlaps,
    _compute_representative_points,
    _filter_pairs_by_com,
    _filter_pairs_touching_boundary,
    _find_axis_contacts,
    _find_contact_center,
    _find_contacts,
    _find_merger_segment_ids,
    _find_small_segment_ids,
    _find_unclaimed_segment_ids,
    _get_unvisited_neighbors,
    _sample_sphere_voxels,
    _voxel_closest_to_mean,
)

# --- Unit tests for helper functions ---


def test_find_axis_contacts_basic():
    """Test finding contacts along one axis."""
    # Two segments touching along x axis
    seg_lo = np.array([[[1, 1]], [[1, 1]]], dtype=np.int64)
    seg_hi = np.array([[[2, 2]], [[2, 2]]], dtype=np.int64)
    aff = np.ones((2, 1, 2), dtype=np.float32) * 0.8

    seg_a, seg_b, aff_vals, _x, _y, _z = _find_axis_contacts(
        seg_lo, seg_hi, aff, offset=(0.5, 0, 0)
    )

    assert len(seg_a) == 4
    assert set(seg_a) == {1}
    assert set(seg_b) == {2}
    np.testing.assert_array_almost_equal(aff_vals, [0.8, 0.8, 0.8, 0.8])


def test_find_axis_contacts_no_contacts():
    """Test no contacts when segments are identical."""
    seg = np.array([[[1, 1]], [[1, 1]]], dtype=np.int64)
    aff = np.ones((2, 1, 2), dtype=np.float32)

    seg_a, _seg_b, _aff_vals, _x, _y, _z = _find_axis_contacts(seg, seg, aff, offset=(0, 0, 0))

    assert len(seg_a) == 0


def test_find_axis_contacts_ignores_zero():
    """Test that contacts with segment 0 are ignored."""
    seg_lo = np.array([[[0, 1]]], dtype=np.int64)
    seg_hi = np.array([[[1, 2]]], dtype=np.int64)
    aff = np.ones((1, 1, 2), dtype=np.float32)

    seg_a, seg_b, _aff_vals, _x, _y, _z = _find_axis_contacts(
        seg_lo, seg_hi, aff, offset=(0, 0, 0)
    )

    # Only (1, 2) contact should be found, not (0, 1)
    assert len(seg_a) == 1
    assert seg_a[0] == 1
    assert seg_b[0] == 2


def test_find_contacts_normalizes_order():
    """Test that segment order is normalized (seg_a < seg_b)."""
    # Create data where seg_b > seg_a in raw data
    seg = np.array([[[1, 2, 1]]], dtype=np.int64)
    aff = np.zeros((3, 1, 1, 3), dtype=np.float32)
    aff[2] = 0.5  # z-axis affinity

    seg_a, seg_b, _aff_vals, _x, _y, _z = _find_contacts(seg, aff, Vec3D(0, 0, 0))

    # All pairs should have seg_a < seg_b
    assert all(a < b for a, b in zip(seg_a, seg_b))


def test_filter_pairs_touching_boundary():
    """Test filtering contacts touching padded boundary."""
    seg_a = np.array([1, 1, 2], dtype=np.int64)
    seg_b = np.array([2, 2, 3], dtype=np.int64)
    aff = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    x = np.array([0.0, 10.0, 10.0], dtype=np.float32)  # 0 is on boundary
    y = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    z = np.array([10.0, 10.0, 10.0], dtype=np.float32)

    shape = (20, 20, 20)
    start = Vec3D(0, 0, 0)

    result = _filter_pairs_touching_boundary(seg_a, seg_b, aff, x, y, z, start, shape)
    seg_a_f, seg_b_f, _aff_f, _x_f, _y_f, _z_f = result

    # Pair (1, 2) has contact at x=0 which is on boundary
    # So only (2, 3) should remain
    assert len(seg_a_f) == 1
    assert seg_a_f[0] == 2
    assert seg_b_f[0] == 3


def test_filter_pairs_by_com():
    """Test filtering contacts by COM outside kernel region."""
    # Pair (1, 2) has contacts at x=4 and x=6, COM at x=5 which is on kernel boundary
    # Pair (2, 3) has contact at x=10 which is inside kernel
    seg_a = np.array([1, 1, 2], dtype=np.int64)
    seg_b = np.array([2, 2, 3], dtype=np.int64)
    aff = np.array([0.5, 0.5, 0.7], dtype=np.float32)
    x = np.array([4.0, 6.0, 10.0], dtype=np.float32)
    y = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    z = np.array([10.0, 10.0, 10.0], dtype=np.float32)

    crop_pad = (5, 5, 5)
    shape = (20, 20, 20)
    start = Vec3D(0, 0, 0)

    result = _filter_pairs_by_com(seg_a, seg_b, aff, x, y, z, start, shape, crop_pad)
    seg_a_f, _seg_b_f, _aff_f, _x_f, _y_f, _z_f = result

    # Pair (1, 2) has COM at x=5 which is exactly on kernel start boundary (included)
    # Pair (2, 3) has COM at x=10 which is inside kernel
    # Both should remain
    assert len(seg_a_f) == 3


def test_filter_pairs_by_com_outside():
    """Test filtering contacts by COM outside kernel region."""
    # Pair (1, 2) has contacts at x=2 and x=4, COM at x=3 which is outside kernel
    seg_a = np.array([1, 1], dtype=np.int64)
    seg_b = np.array([2, 2], dtype=np.int64)
    aff = np.array([0.5, 0.5], dtype=np.float32)
    x = np.array([2.0, 4.0], dtype=np.float32)
    y = np.array([10.0, 10.0], dtype=np.float32)
    z = np.array([10.0, 10.0], dtype=np.float32)

    crop_pad = (5, 5, 5)
    shape = (20, 20, 20)
    start = Vec3D(0, 0, 0)

    result = _filter_pairs_by_com(seg_a, seg_b, aff, x, y, z, start, shape, crop_pad)
    seg_a_f, _seg_b_f, _aff_f, _x_f, _y_f, _z_f = result

    # Pair (1, 2) has COM at x=3 which is outside kernel (5-15)
    assert len(seg_a_f) == 0


def test_compute_overlaps_basic():
    """Test computing overlaps between segments and reference."""
    seg = np.array([[[1, 1, 2], [1, 1, 2]]], dtype=np.int64)
    ref = np.array([[[1, 1, 1], [1, 1, 2]]], dtype=np.int64)

    seg_ids, _ref_ids, _counts = _compute_overlaps(seg, ref)

    # Segment 1 overlaps ref 1 (4 voxels)
    # Segment 2 overlaps ref 1 (1 voxel) and ref 2 (1 voxel)
    assert len(seg_ids) > 0


def test_find_small_segment_ids():
    """Test finding segments below size threshold."""
    seg = np.zeros((10, 10, 10), dtype=np.int64)
    seg[:5, :, :] = 1  # 500 voxels
    seg[5:6, :, :] = 2  # 100 voxels
    seg[6:, :, :] = 3  # 400 voxels

    small_ids = _find_small_segment_ids(seg, min_seg_size_vx=200)

    assert 2 in small_ids
    assert 1 not in small_ids
    assert 3 not in small_ids


def test_find_merger_segment_ids():
    """Test finding segments that overlap multiple reference CCs."""
    seg_ids = np.array([1, 1, 2, 2], dtype=np.int64)
    ref_ids = np.array([10, 20, 30, 30], dtype=np.int64)
    counts = np.array([100, 100, 100, 100], dtype=np.int32)

    merger_ids = _find_merger_segment_ids(seg_ids, ref_ids, counts, min_overlap_vx=50)

    # Segment 1 overlaps ref 10 and 20 -> merger
    # Segment 2 overlaps only ref 30 -> not merger
    assert 1 in merger_ids
    assert 2 not in merger_ids


def test_find_unclaimed_segment_ids():
    """Test finding segments without sufficient overlap."""
    seg_ids = np.array([1, 2, 3], dtype=np.int64)
    counts = np.array([100, 50, 10], dtype=np.int32)
    all_seg_ids = {1, 2, 3}

    unclaimed = _find_unclaimed_segment_ids(
        seg_ids, counts, min_overlap_vx=60, all_seg_ids=all_seg_ids
    )

    assert 2 in unclaimed
    assert 3 in unclaimed
    assert 1 not in unclaimed


def test_build_seg_to_ref():
    """Test building segment to reference mapping."""
    seg_ids = np.array([1, 1, 2], dtype=np.int64)
    ref_ids = np.array([10, 20, 30], dtype=np.int64)
    counts = np.array([100, 50, 100], dtype=np.int32)

    seg_to_ref = _build_seg_to_ref(seg_ids, ref_ids, counts, min_overlap_vx=60)

    assert seg_to_ref[1] == {10}  # 20 filtered out due to low count
    assert seg_to_ref[2] == {30}


def test_blackout_segments():
    """Test setting segment IDs to 0."""
    seg = np.array([[[1, 2, 3], [1, 2, 3]]], dtype=np.int64)
    result = _blackout_segments(seg, {2, 3})

    assert np.all(result[seg == 1] == 1)
    assert np.all(result[seg == 2] == 0)
    assert np.all(result[seg == 3] == 0)


def test_blackout_segments_empty():
    """Test blackout with empty set does nothing."""
    seg = np.array([[[1, 2, 3]]], dtype=np.int64)
    result = _blackout_segments(seg, set())

    np.testing.assert_array_equal(result, seg)


def test_compute_contact_counts():
    """Test counting contacts per segment pair."""
    seg_a = np.array([1, 1, 2, 2, 2], dtype=np.int64)
    seg_b = np.array([3, 3, 4, 4, 4], dtype=np.int64)

    counts = _compute_contact_counts(seg_a, seg_b)

    assert counts[(1, 3)] == 2
    assert counts[(2, 4)] == 3


def test_compute_affinity_weighted_com():
    """Test affinity-weighted center of mass computation."""
    contacts = [(0.0, 0.0, 0.0, 0.9), (10.0, 0.0, 0.0, 0.1)]
    resolution = np.array([16.0, 16.0, 40.0])

    com = _compute_affinity_weighted_com(contacts, resolution)

    # COM should be closer to x=0 due to higher affinity weight
    assert com[0] < 5.0 * 16.0  # Would be 5.0 * 16 = 80 if unweighted


def test_compute_affinity_weighted_com_zero_affinity():
    """Test COM computation when all affinities are zero."""
    contacts = [(0.0, 0.0, 0.0, 0.0), (10.0, 0.0, 0.0, 0.0)]
    resolution = np.array([16.0, 16.0, 40.0])

    com = _compute_affinity_weighted_com(contacts, resolution)

    # Should fall back to simple mean
    np.testing.assert_array_almost_equal(com, [5.0 * 16.0, 0.0, 0.0])


# --- Representative points helper function tests ---


def test_build_voxel_spatial_hash():
    """Test building spatial hash from voxel coordinates."""
    voxels = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 0]], dtype=np.int64)
    result = _build_voxel_spatial_hash(voxels)

    assert (0, 0, 0) in result
    assert (0, 0, 1) in result
    assert (1, 0, 0) in result
    assert len(result[(0, 0, 0)]) == 2  # indices 0 and 3
    assert 0 in result[(0, 0, 0)]
    assert 3 in result[(0, 0, 0)]


def test_get_unvisited_neighbors():
    """Test getting unvisited neighbors using 6-connectivity."""
    coord_to_indices = {
        (5, 5, 5): [0],
        (6, 5, 5): [1],  # neighbor +x
        (4, 5, 5): [2],  # neighbor -x
        (5, 6, 5): [3],  # neighbor +y
        (7, 5, 5): [4],  # not a neighbor (2 away)
    }
    visited = np.array([False, False, True, False, False])

    neighbors = _get_unvisited_neighbors((5, 5, 5), coord_to_indices, visited)

    # Should get indices 0 (same cell), 1 (+x neighbor), 3 (+y neighbor)
    # Index 2 is visited, index 4 is not a neighbor
    assert 0 in neighbors  # same cell
    assert 1 in neighbors  # +x neighbor
    assert 3 in neighbors  # +y neighbor
    assert 2 not in neighbors  # visited
    assert 4 not in neighbors  # not adjacent


def test_compute_contact_connected_components_single():
    """Test connected components with single component."""
    resolution = np.array([16.0, 16.0, 40.0])
    # 3 adjacent voxels in a line along x
    contact_faces = np.array(
        [
            [0.0, 0.0, 0.0, 0.5],
            [16.0, 0.0, 0.0, 0.5],
            [32.0, 0.0, 0.0, 0.5],
        ],
        dtype=np.float32,
    )

    components = _compute_contact_connected_components(contact_faces, resolution)

    assert len(components) == 1
    assert len(components[0]) == 3


def test_compute_contact_connected_components_multiple():
    """Test connected components with multiple disjoint components."""
    resolution = np.array([16.0, 16.0, 40.0])
    # Two disjoint pairs of voxels - must be >1 voxel apart
    contact_faces = np.array(
        [
            [0.0, 0.0, 0.0, 0.5],  # component 1, voxel (0,0,0)
            [16.0, 0.0, 0.0, 0.5],  # component 1, voxel (1,0,0)
            [160.0, 0.0, 0.0, 0.5],  # component 2 (far away), voxel (10,0,0)
            [176.0, 0.0, 0.0, 0.5],  # component 2, voxel (11,0,0)
        ],
        dtype=np.float32,
    )

    components = _compute_contact_connected_components(contact_faces, resolution)

    assert len(components) == 2
    sizes = sorted([len(c) for c in components])
    assert sizes == [2, 2]


def test_compute_contact_connected_components_empty():
    """Test connected components with empty input."""
    resolution = np.array([16.0, 16.0, 40.0])
    contact_faces = np.array([], dtype=np.float32).reshape(0, 4)

    components = _compute_contact_connected_components(contact_faces, resolution)

    assert len(components) == 0


def test_find_contact_center():
    """Test finding contact center from largest component."""
    resolution = np.array([16.0, 16.0, 40.0])
    # Larger component at origin, smaller component far away
    contact_faces = np.array(
        [
            [0.0, 0.0, 0.0, 0.5],
            [16.0, 0.0, 0.0, 0.5],
            [32.0, 0.0, 0.0, 0.5],
            [1000.0, 1000.0, 1000.0, 0.5],  # isolated point
        ],
        dtype=np.float32,
    )

    center = _find_contact_center(contact_faces, resolution)

    # Center should be from the larger component (around x=16)
    assert center[0] == 16.0  # middle of the 3-point component


def test_sample_sphere_voxels():
    """Test sampling voxels within sphere."""
    seg_volume = np.zeros((10, 10, 10), dtype=np.int64)
    seg_volume[5, 5, 5] = 1
    seg_volume[4, 5, 5] = 1
    seg_volume[6, 5, 5] = 2
    resolution = np.array([16.0, 16.0, 40.0])
    seg_start_nm = np.array([0.0, 0.0, 0.0])
    center_nm = np.array([80.0, 80.0, 200.0])  # center at voxel (5, 5, 5)

    local_vx, voxel_nm_result, seg_ids_result, _ = _sample_sphere_voxels(
        center_nm, 50.0, seg_volume, seg_start_nm, resolution
    )

    assert len(local_vx) > 0
    assert len(voxel_nm_result) == len(local_vx)
    assert len(seg_ids_result) == len(local_vx)


def test_sample_sphere_voxels_empty():
    """Test sampling voxels outside volume returns empty."""
    seg_volume = np.zeros((10, 10, 10), dtype=np.int64)
    resolution = np.array([16.0, 16.0, 40.0])
    seg_start_nm = np.array([0.0, 0.0, 0.0])
    center_nm = np.array([10000.0, 10000.0, 10000.0])  # far outside volume

    local_vx, _, _, _ = _sample_sphere_voxels(
        center_nm, 50.0, seg_volume, seg_start_nm, resolution
    )

    assert len(local_vx) == 0


def test_voxel_closest_to_mean():
    """Test finding voxel closest to mean."""
    voxel_nm = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mask = np.array([True, True, True])

    result = _voxel_closest_to_mean(voxel_nm, mask)

    # Mean is (50, 0, 0), closest voxel is index 2
    assert result is not None
    assert result[0] == 50.0
    assert result[1] == 0.0
    assert result[2] == 0.0


def test_voxel_closest_to_mean_empty_mask():
    """Test finding voxel with empty mask returns None."""
    voxel_nm = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32)
    mask = np.array([False, False])

    result = _voxel_closest_to_mean(voxel_nm, mask)

    assert result is None


def test_compute_representative_points():
    """Test computing representative points."""
    # Create a simple segmentation volume
    seg_volume = np.zeros((20, 20, 20), dtype=np.int64)
    seg_volume[8:12, 8:12, 8:12] = 1  # segment 1 in center-left
    seg_volume[12:16, 8:12, 8:12] = 2  # segment 2 in center-right

    resolution = np.array([16.0, 16.0, 40.0])
    seg_start_nm = np.array([0.0, 0.0, 0.0])

    # Contact faces at the boundary between segments (x=12)
    contact_faces = np.array(
        [
            [192.0, 160.0, 400.0, 0.5],  # x=12 in nm
            [192.0, 176.0, 400.0, 0.5],
        ],
        dtype=np.float32,
    )

    result = _compute_representative_points(
        contact_faces, 1, 2, seg_volume, seg_start_nm, resolution
    )

    assert 1 in result
    assert 2 in result
    # Points should be Vec3D instances
    assert hasattr(result[1], "__getitem__")
    assert hasattr(result[2], "__getitem__")


def test_compute_representative_points_fallback():
    """Test representative points fallback when segments not in sphere."""
    # Create segmentation with segments far from contact
    seg_volume = np.zeros((20, 20, 20), dtype=np.int64)
    seg_volume[0, 0, 0] = 1
    seg_volume[19, 19, 19] = 2

    resolution = np.array([16.0, 16.0, 40.0])
    seg_start_nm = np.array([0.0, 0.0, 0.0])

    # Contact faces in the middle (no segments in 200nm sphere)
    contact_faces = np.array(
        [
            [160.0, 160.0, 400.0, 0.5],
        ],
        dtype=np.float32,
    )

    result = _compute_representative_points(
        contact_faces, 1, 2, seg_volume, seg_start_nm, resolution
    )

    # Should fall back to contact center for both
    assert 1 in result
    assert 2 in result
    # Both should be at or near contact center (160, 160, 400)
    assert result[1][0] == 160.0
    assert result[2][0] == 160.0


# --- SegContactOp method tests ---


def test_seg_contact_op_with_added_crop_pad():
    """Test with_added_crop_pad method."""
    op = SegContactOp(crop_pad=(10, 10, 10))
    op2 = op.with_added_crop_pad(Vec3D(5, 5, 5))

    assert tuple(op2.crop_pad) == (15, 15, 15)


def test_seg_contact_op_get_input_resolution():
    """Test get_input_resolution returns same resolution."""
    op = SegContactOp()
    res = Vec3D(16.0, 16.0, 40.0)

    assert op.get_input_resolution(res) == res

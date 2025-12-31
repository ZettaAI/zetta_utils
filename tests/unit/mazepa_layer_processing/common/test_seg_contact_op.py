import numpy as np

from zetta_utils.geometry import Vec3D
from zetta_utils.mazepa_layer_processing.common.seg_contact_op import (
    SegContactOp,
    _blackout_segments,
    _build_seg_to_ref,
    _compute_affinity_weighted_com,
    _compute_contact_counts,
    _compute_overlaps,
    _filter_pairs_to_kernel,
    _find_axis_contacts,
    _find_contacts,
    _find_merger_segment_ids,
    _find_small_segment_ids,
    _find_unclaimed_segment_ids,
)


# --- Unit tests for helper functions ---


def test_find_axis_contacts_basic():
    """Test finding contacts along one axis."""
    # Two segments touching along x axis
    seg_lo = np.array([[[1, 1]], [[1, 1]]], dtype=np.int64)
    seg_hi = np.array([[[2, 2]], [[2, 2]]], dtype=np.int64)
    aff = np.ones((2, 1, 2), dtype=np.float32) * 0.8

    seg_a, seg_b, aff_vals, x, y, z = _find_axis_contacts(
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

    seg_a, seg_b, aff_vals, x, y, z = _find_axis_contacts(seg, seg, aff, offset=(0, 0, 0))

    assert len(seg_a) == 0


def test_find_axis_contacts_ignores_zero():
    """Test that contacts with segment 0 are ignored."""
    seg_lo = np.array([[[0, 1]]], dtype=np.int64)
    seg_hi = np.array([[[1, 2]]], dtype=np.int64)
    aff = np.ones((1, 1, 2), dtype=np.float32)

    seg_a, seg_b, aff_vals, x, y, z = _find_axis_contacts(seg_lo, seg_hi, aff, offset=(0, 0, 0))

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

    seg_a, seg_b, aff_vals, x, y, z = _find_contacts(seg, aff, Vec3D(0, 0, 0))

    # All pairs should have seg_a < seg_b
    assert all(a < b for a, b in zip(seg_a, seg_b))


def test_filter_pairs_to_kernel():
    """Test filtering contacts to kernel region."""
    seg_a = np.array([1, 1, 2], dtype=np.int64)
    seg_b = np.array([2, 2, 3], dtype=np.int64)
    aff = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    x = np.array([5.0, 15.0, 10.0], dtype=np.float32)  # 5 outside, 15 outside
    y = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    z = np.array([10.0, 10.0, 10.0], dtype=np.float32)

    crop_pad = (5, 5, 5)
    shape = (20, 20, 20)
    start = Vec3D(0, 0, 0)

    result = _filter_pairs_to_kernel(seg_a, seg_b, aff, x, y, z, start, shape, crop_pad)
    seg_a_f, seg_b_f, aff_f, x_f, y_f, z_f = result

    # Pair (1, 2) has contacts at x=5 and x=15 which are outside kernel (5-15)
    # So only (2, 3) should remain
    assert len(seg_a_f) == 1
    assert seg_a_f[0] == 2
    assert seg_b_f[0] == 3


def test_compute_overlaps_basic():
    """Test computing overlaps between segments and reference."""
    seg = np.array([[[1, 1, 2], [1, 1, 2]]], dtype=np.int64)
    ref = np.array([[[1, 1, 1], [1, 1, 2]]], dtype=np.int64)

    seg_ids, ref_ids, counts = _compute_overlaps(seg, ref)

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

    unclaimed = _find_unclaimed_segment_ids(seg_ids, counts, min_overlap_vx=60)

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


# --- SegContactOp method tests ---


def test_seg_contact_op_with_added_crop_pad():
    """Test with_added_crop_pad method."""
    op = SegContactOp(sphere_radius_nm=1000.0, crop_pad=(10, 10, 10))
    op2 = op.with_added_crop_pad(Vec3D(5, 5, 5))

    assert tuple(op2.crop_pad) == (15, 15, 15)


def test_seg_contact_op_get_input_resolution():
    """Test get_input_resolution returns same resolution."""
    op = SegContactOp(sphere_radius_nm=1000.0)
    res = Vec3D(16.0, 16.0, 40.0)

    assert op.get_input_resolution(res) == res

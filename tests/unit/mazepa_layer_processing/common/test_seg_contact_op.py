import tempfile

import numpy as np

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.seg_contact import (
    SegContactLayerBackend,
    VolumetricSegContactLayer,
)
from zetta_utils.mazepa_layer_processing.common.seg_contact_op import (
    SegContactOp,
    _build_seg_contacts,
    _filter_pairs_to_kernel,
    _find_axis_contacts,
    _find_contacts,
)


class MockVolumetricLayer:
    """Mock layer that returns predefined data."""

    def __init__(self, data: np.ndarray, resolution: Vec3D = Vec3D(16, 16, 40)):
        self.data = data
        self.resolution = resolution

    def __getitem__(self, idx: VolumetricIndex) -> np.ndarray:
        return self.data


def make_backend(temp_dir: str) -> SegContactLayerBackend:
    """Helper to create a backend for testing."""
    backend = SegContactLayerBackend(
        path=temp_dir,
        resolution=Vec3D(16, 16, 40),
        voxel_offset=Vec3D(0, 0, 0),
        size=Vec3D(100, 100, 100),
        chunk_size=Vec3D(32, 32, 32),
        max_contact_span=64,
    )
    backend.write_info()
    return backend


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

    seg_a, seg_b, aff_vals, x, y, z = _find_axis_contacts(
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

    # Pair (1, 2) has contact at x=5 which is outside kernel (5-15)
    # So only (2, 3) should remain
    assert len(seg_a_f) == 1
    assert seg_a_f[0] == 2
    assert seg_b_f[0] == 3


def test_build_seg_contacts_basic():
    """Test building SegContact objects from raw data."""
    seg_a = np.array([1, 1, 1], dtype=np.int64)
    seg_b = np.array([2, 2, 2], dtype=np.int64)
    aff = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    x = np.array([10.0, 11.0, 12.0], dtype=np.float32)
    y = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    z = np.array([30.0, 30.0, 30.0], dtype=np.float32)

    resolution = Vec3D(16, 16, 40)
    contacts = _build_seg_contacts(
        seg_a, seg_b, aff, x, y, z, resolution, min_contact_vx=1, max_contact_vx=100
    )

    assert len(contacts) == 1
    c = contacts[0]
    assert c.seg_a == 1
    assert c.seg_b == 2
    assert c.contact_faces.shape == (3, 4)


def test_build_seg_contacts_filters_by_count():
    """Test that contacts are filtered by min/max count."""
    # 3 contacts for pair (1, 2), 1 contact for pair (3, 4)
    seg_a = np.array([1, 1, 1, 3], dtype=np.int64)
    seg_b = np.array([2, 2, 2, 4], dtype=np.int64)
    aff = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    x = np.array([10.0, 11.0, 12.0, 20.0], dtype=np.float32)
    y = np.array([20.0, 20.0, 20.0, 20.0], dtype=np.float32)
    z = np.array([30.0, 30.0, 30.0, 30.0], dtype=np.float32)

    resolution = Vec3D(16, 16, 40)

    # min=2 should filter out (3, 4)
    contacts = _build_seg_contacts(
        seg_a, seg_b, aff, x, y, z, resolution, min_contact_vx=2, max_contact_vx=100
    )
    assert len(contacts) == 1
    assert contacts[0].seg_a == 1

    # max=2 should filter out (1, 2)
    contacts = _build_seg_contacts(
        seg_a, seg_b, aff, x, y, z, resolution, min_contact_vx=1, max_contact_vx=2
    )
    assert len(contacts) == 1
    assert contacts[0].seg_a == 3


def test_build_seg_contacts_affinity_weighted_com():
    """Test that COM is affinity-weighted."""
    seg_a = np.array([1, 1], dtype=np.int64)
    seg_b = np.array([2, 2], dtype=np.int64)
    aff = np.array([0.9, 0.1], dtype=np.float32)  # First has much higher weight
    x = np.array([0.0, 10.0], dtype=np.float32)
    y = np.array([0.0, 0.0], dtype=np.float32)
    z = np.array([0.0, 0.0], dtype=np.float32)

    resolution = Vec3D(1, 1, 1)
    contacts = _build_seg_contacts(
        seg_a, seg_b, aff, x, y, z, resolution, min_contact_vx=1, max_contact_vx=100
    )

    # COM should be closer to x=0 due to higher affinity weight
    assert contacts[0].com[0] < 5.0  # Would be 5.0 if unweighted


# --- Integration tests ---


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


def test_seg_contact_op_call_writes_contacts():
    """Test that __call__ writes contacts to the layer."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend(temp_dir)
        layer = VolumetricSegContactLayer(backend=backend)

        # Create segmentation with two touching segments
        seg = np.zeros((10, 10, 10), dtype=np.int64)
        seg[:5, :, :] = 1
        seg[5:, :, :] = 2

        # Create affinity data (3 channels for x, y, z)
        aff = np.ones((3, 10, 10, 10), dtype=np.float32) * 0.8

        resolution = Vec3D(16, 16, 40)
        seg_layer = MockVolumetricLayer(seg, resolution)
        aff_layer = MockVolumetricLayer(aff, resolution)

        # Create idx in voxel coordinates that aligns with resolution
        idx = VolumetricIndex(
            resolution=resolution,
            bbox=BBox3D.from_coords(
                start_coord=[0, 0, 0], end_coord=[10, 10, 10], resolution=resolution
            ),
        )

        op = SegContactOp(min_contact_vx=1, max_contact_vx=1000)
        op(idx, layer, seg_layer, aff_layer)

        # Read back and verify
        result = layer[idx]
        assert len(result) == 1
        assert result[0].seg_a == 1
        assert result[0].seg_b == 2


def test_seg_contact_op_call_empty_result():
    """Test that __call__ handles no contacts gracefully."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend(temp_dir)
        layer = VolumetricSegContactLayer(backend=backend)

        # Single segment - no contacts
        seg = np.ones((10, 10, 10), dtype=np.int64)
        aff = np.ones((3, 10, 10, 10), dtype=np.float32)

        resolution = Vec3D(16, 16, 40)
        seg_layer = MockVolumetricLayer(seg, resolution)
        aff_layer = MockVolumetricLayer(aff, resolution)

        idx = VolumetricIndex(
            resolution=resolution,
            bbox=BBox3D.from_coords(
                start_coord=[0, 0, 0], end_coord=[10, 10, 10], resolution=resolution
            ),
        )

        op = SegContactOp()
        op(idx, layer, seg_layer, aff_layer)

        result = layer[idx]
        assert len(result) == 0


def test_seg_contact_op_call_multiple_pairs():
    """Test with multiple segment pairs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend(temp_dir)
        layer = VolumetricSegContactLayer(backend=backend)

        # Three segments: 1, 2, 3 touching in sequence
        seg = np.zeros((12, 4, 4), dtype=np.int64)
        seg[:4, :, :] = 1
        seg[4:8, :, :] = 2
        seg[8:, :, :] = 3

        aff = np.ones((3, 12, 4, 4), dtype=np.float32) * 0.7

        resolution = Vec3D(16, 16, 40)
        seg_layer = MockVolumetricLayer(seg, resolution)
        aff_layer = MockVolumetricLayer(aff, resolution)

        idx = VolumetricIndex(
            resolution=resolution,
            bbox=BBox3D.from_coords(
                start_coord=[0, 0, 0], end_coord=[12, 4, 4], resolution=resolution
            ),
        )

        op = SegContactOp(min_contact_vx=1, max_contact_vx=1000)
        op(idx, layer, seg_layer, aff_layer)

        result = layer[idx]
        # Should have 2 contact pairs: (1, 2) and (2, 3)
        assert len(result) == 2
        pairs = {(c.seg_a, c.seg_b) for c in result}
        assert pairs == {(1, 2), (2, 3)}

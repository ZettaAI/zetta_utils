import tempfile

import numpy as np
import pytest

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.seg_contact import (
    SegContact,
    SegContactLayerBackend,
    VolumetricSegContactLayer,
)


def make_backend(temp_dir: str) -> SegContactLayerBackend:
    """Helper to create a backend for testing."""
    backend = SegContactLayerBackend(
        path=temp_dir,
        resolution=Vec3D(16, 16, 40),
        voxel_offset=Vec3D(0, 0, 0),
        size=Vec3D(1000, 1000, 500),
        chunk_size=Vec3D(256, 256, 128),
        max_contact_span=512,
    )
    backend.write_info()
    return backend


def make_seg_contact(
    id: int = 1,
    seg_a: int = 100,
    seg_b: int = 200,
    com: tuple[float, float, float] = (100.0, 100.0, 100.0),
    n_faces: int = 3,
) -> SegContact:
    """Helper to create a SegContact for testing."""
    contact_faces = np.array(
        [[com[0] + i, com[1] + i, com[2] + i, 0.5] for i in range(n_faces)],
        dtype=np.float32,
    )
    return SegContact(
        id=id,
        seg_a=seg_a,
        seg_b=seg_b,
        com=Vec3D(*com),
        contact_faces=contact_faces,
    )


# --- Basic instantiation tests ---


def test_layer_instantiation():
    """Test basic layer creation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend(temp_dir)
        layer = VolumetricSegContactLayer(backend=backend)

        assert layer.backend is backend
        assert layer.readonly is False


def test_layer_instantiation_readonly():
    """Test layer creation with readonly=True."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend(temp_dir)
        layer = VolumetricSegContactLayer(backend=backend, readonly=True)

        assert layer.readonly is True


# --- Read tests ---


def test_getitem_empty():
    """Test reading from empty layer."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend(temp_dir)
        layer = VolumetricSegContactLayer(backend=backend)

        idx = VolumetricIndex(
            resolution=Vec3D(16, 16, 40),
            bbox=BBox3D.from_slices((slice(0, 256), slice(0, 256), slice(0, 128))),
        )
        result = layer[idx]

        assert len(result) == 0


def test_getitem_single_contact():
    """Test reading a single contact."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend(temp_dir)
        contact = make_seg_contact(id=42, com=(100.0, 100.0, 100.0))
        backend.write_chunk((0, 0, 0), [contact])

        layer = VolumetricSegContactLayer(backend=backend)

        idx = VolumetricIndex(
            resolution=Vec3D(16, 16, 40),
            bbox=BBox3D.from_slices((slice(0, 256), slice(0, 256), slice(0, 128))),
        )
        result = layer[idx]

        assert len(result) == 1
        assert result[0].id == 42


def test_getitem_multiple_contacts():
    """Test reading multiple contacts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend(temp_dir)
        contacts = [
            make_seg_contact(id=1, com=(100.0, 100.0, 100.0)),
            make_seg_contact(id=2, com=(200.0, 200.0, 200.0)),
            make_seg_contact(id=3, com=(300.0, 300.0, 300.0)),
        ]
        backend.write_chunk((0, 0, 0), contacts)

        layer = VolumetricSegContactLayer(backend=backend)

        idx = VolumetricIndex(
            resolution=Vec3D(16, 16, 40),
            bbox=BBox3D.from_slices((slice(0, 256), slice(0, 256), slice(0, 128))),
        )
        result = layer[idx]

        assert len(result) == 3
        ids = {c.id for c in result}
        assert ids == {1, 2, 3}


def test_getitem_filters_by_bbox():
    """Test that getitem filters contacts by bbox."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend(temp_dir)
        # All in first chunk but different positions
        contacts = [
            make_seg_contact(id=1, com=(100.0, 100.0, 100.0)),  # in query
            make_seg_contact(id=2, com=(2000.0, 100.0, 100.0)),  # outside query in x
        ]
        backend.write_chunk((0, 0, 0), contacts)

        layer = VolumetricSegContactLayer(backend=backend)

        # Query only first part (0-1000 nm in x = 0-62.5 voxels at 16nm)
        idx = VolumetricIndex(
            resolution=Vec3D(16, 16, 40),
            bbox=BBox3D.from_slices((slice(0, 64), slice(0, 256), slice(0, 128))),
        )
        result = layer[idx]

        assert len(result) == 1
        assert result[0].id == 1


# --- Write tests ---


def test_setitem_single_contact():
    """Test writing a single contact."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend(temp_dir)
        layer = VolumetricSegContactLayer(backend=backend)

        contact = make_seg_contact(id=42, com=(100.0, 100.0, 100.0))

        idx = VolumetricIndex(
            resolution=Vec3D(16, 16, 40),
            bbox=BBox3D.from_slices((slice(0, 256), slice(0, 256), slice(0, 128))),
        )
        layer[idx] = [contact]

        # Read back
        result = layer[idx]
        assert len(result) == 1
        assert result[0].id == 42


def test_setitem_multiple_contacts():
    """Test writing multiple contacts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend(temp_dir)
        layer = VolumetricSegContactLayer(backend=backend)

        contacts = [
            make_seg_contact(id=1, com=(100.0, 100.0, 100.0)),
            make_seg_contact(id=2, com=(200.0, 200.0, 200.0)),
        ]

        idx = VolumetricIndex(
            resolution=Vec3D(16, 16, 40),
            bbox=BBox3D.from_slices((slice(0, 256), slice(0, 256), slice(0, 128))),
        )
        layer[idx] = contacts

        result = layer[idx]
        assert len(result) == 2


def test_setitem_readonly_raises():
    """Test that writing to readonly layer raises."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend(temp_dir)
        layer = VolumetricSegContactLayer(backend=backend, readonly=True)

        contact = make_seg_contact()
        idx = VolumetricIndex(
            resolution=Vec3D(16, 16, 40),
            bbox=BBox3D.from_slices((slice(0, 256), slice(0, 256), slice(0, 128))),
        )

        with pytest.raises(Exception):  # Could be IOError, PermissionError, etc.
            layer[idx] = [contact]


def test_setitem_distributes_to_chunks():
    """Test that writing distributes contacts to correct chunks."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend(temp_dir)
        layer = VolumetricSegContactLayer(backend=backend)

        # Contact 1 in chunk (0,0,0), Contact 2 in chunk (1,0,0)
        # chunk_size_nm = 256 * 16 = 4096
        contacts = [
            make_seg_contact(id=1, com=(100.0, 100.0, 100.0)),
            make_seg_contact(id=2, com=(5000.0, 100.0, 100.0)),  # > 4096
        ]

        idx = VolumetricIndex(
            resolution=Vec3D(16, 16, 40),
            bbox=BBox3D.from_slices((slice(0, 512), slice(0, 256), slice(0, 128))),
        )
        layer[idx] = contacts

        # Check chunks directly
        chunk_0 = backend.read_chunk((0, 0, 0))
        chunk_1 = backend.read_chunk((1, 0, 0))

        assert len(chunk_0) == 1
        assert chunk_0[0].id == 1
        assert len(chunk_1) == 1
        assert chunk_1[0].id == 2


# --- Round-trip tests ---


def test_round_trip():
    """Test full round-trip: write then read."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend(temp_dir)
        layer = VolumetricSegContactLayer(backend=backend)

        contacts = [
            make_seg_contact(id=1, seg_a=100, seg_b=200, com=(100.0, 100.0, 100.0), n_faces=5),
            make_seg_contact(id=2, seg_a=100, seg_b=300, com=(200.0, 200.0, 200.0), n_faces=10),
        ]

        idx = VolumetricIndex(
            resolution=Vec3D(16, 16, 40),
            bbox=BBox3D.from_slices((slice(0, 256), slice(0, 256), slice(0, 128))),
        )
        layer[idx] = contacts

        result = layer[idx]

        assert len(result) == 2
        result_by_id = {c.id: c for c in result}

        for orig in contacts:
            read = result_by_id[orig.id]
            assert read.seg_a == orig.seg_a
            assert read.seg_b == orig.seg_b
            assert read.com == orig.com
            np.testing.assert_array_almost_equal(read.contact_faces, orig.contact_faces)

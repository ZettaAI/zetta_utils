import json
import os
import tempfile

import numpy as np
import pytest

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.seg_contact import SegContact, SegContactLayerBackend

# --- Chunk naming tests ---


def test_get_chunk_name():
    """Test chunk naming follows precomputed format."""
    backend = SegContactLayerBackend(
        path="/tmp/test",
        resolution=Vec3D(16, 16, 40),
        voxel_offset=Vec3D(0, 0, 0),
        size=Vec3D(1000, 1000, 500),
        chunk_size=Vec3D(256, 256, 128),
        max_contact_span=512,
    )

    # First chunk at origin
    assert backend.get_chunk_name((0, 0, 0)) == "0-256_0-256_0-128"

    # Second chunk in x
    assert backend.get_chunk_name((1, 0, 0)) == "256-512_0-256_0-128"

    # Chunk at (1, 2, 3)
    assert backend.get_chunk_name((1, 2, 3)) == "256-512_512-768_384-512"


def test_get_chunk_name_with_offset():
    """Test chunk naming with non-zero voxel offset."""
    backend = SegContactLayerBackend(
        path="/tmp/test",
        resolution=Vec3D(16, 16, 40),
        voxel_offset=Vec3D(100, 200, 50),
        size=Vec3D(1000, 1000, 500),
        chunk_size=Vec3D(256, 256, 128),
        max_contact_span=512,
    )

    # First chunk starts at offset
    assert backend.get_chunk_name((0, 0, 0)) == "100-356_200-456_50-178"


def test_get_chunk_path():
    """Test chunk path includes contacts subdirectory."""
    backend = SegContactLayerBackend(
        path="/tmp/test_dataset",
        resolution=Vec3D(16, 16, 40),
        voxel_offset=Vec3D(0, 0, 0),
        size=Vec3D(1000, 1000, 500),
        chunk_size=Vec3D(256, 256, 128),
        max_contact_span=512,
    )

    chunk_path = backend.get_chunk_path((0, 0, 0))
    assert chunk_path == "/tmp/test_dataset/contacts/0-256_0-256_0-128"


# --- COM to chunk index tests ---


def test_com_to_chunk_idx():
    """Test converting COM in nanometers to chunk grid index."""
    backend = SegContactLayerBackend(
        path="/tmp/test",
        resolution=Vec3D(16, 16, 40),
        voxel_offset=Vec3D(0, 0, 0),
        size=Vec3D(1000, 1000, 500),
        chunk_size=Vec3D(256, 256, 128),
        max_contact_span=512,
    )

    # COM at origin -> chunk (0, 0, 0)
    assert backend.com_to_chunk_idx(Vec3D(0.0, 0.0, 0.0)) == (0, 0, 0)

    # COM in middle of first chunk (chunk_size * resolution / 2)
    # chunk_size[0] * resolution[0] = 256 * 16 = 4096 nm
    # So 2000 nm is still in first chunk
    assert backend.com_to_chunk_idx(Vec3D(2000.0, 2000.0, 2000.0)) == (0, 0, 0)

    # COM at start of second chunk in x
    # 256 voxels * 16 nm/voxel = 4096 nm
    assert backend.com_to_chunk_idx(Vec3D(4096.0, 0.0, 0.0)) == (1, 0, 0)


def test_com_to_chunk_idx_with_offset():
    """Test COM to chunk index with non-zero voxel offset."""
    backend = SegContactLayerBackend(
        path="/tmp/test",
        resolution=Vec3D(16, 16, 40),
        voxel_offset=Vec3D(100, 0, 0),  # offset of 100 voxels in x = 1600 nm
        size=Vec3D(1000, 1000, 500),
        chunk_size=Vec3D(256, 256, 128),
        max_contact_span=512,
    )

    # COM at 1600 nm (which is voxel 100, the start) -> chunk (0, 0, 0)
    assert backend.com_to_chunk_idx(Vec3D(1600.0, 0.0, 0.0)) == (0, 0, 0)

    # COM at 1600 + 4096 = 5696 nm -> chunk (1, 0, 0)
    assert backend.com_to_chunk_idx(Vec3D(5696.0, 0.0, 0.0)) == (1, 0, 0)


def test_com_to_chunk_idx_different_resolutions():
    """Test COM to chunk index with anisotropic resolution."""
    backend = SegContactLayerBackend(
        path="/tmp/test",
        resolution=Vec3D(8, 8, 40),  # different z resolution
        voxel_offset=Vec3D(0, 0, 0),
        size=Vec3D(1000, 1000, 500),
        chunk_size=Vec3D(256, 256, 128),
        max_contact_span=512,
    )

    # chunk_size_nm = (256*8, 256*8, 128*40) = (2048, 2048, 5120)
    assert backend.com_to_chunk_idx(Vec3D(0.0, 0.0, 0.0)) == (0, 0, 0)
    assert backend.com_to_chunk_idx(Vec3D(2048.0, 0.0, 0.0)) == (1, 0, 0)
    assert backend.com_to_chunk_idx(Vec3D(0.0, 0.0, 5120.0)) == (0, 0, 1)


# --- Info file tests ---


def test_write_info_creates_file():
    """Test that write_info creates a valid info JSON file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(1000, 1000, 500),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )

        backend.write_info()

        info_path = os.path.join(temp_dir, "info")
        assert os.path.exists(info_path)

        with open(info_path, "r") as f:
            info = json.load(f)

        assert info["type"] == "seg_contact"
        assert info["resolution"] == [16, 16, 40]
        assert info["chunk_size"] == [256, 256, 128]


def test_write_info_all_fields():
    """Test that write_info writes all required fields."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(100, 200, 50),
            size=Vec3D(1000, 1000, 500),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )

        backend.write_info()

        with open(os.path.join(temp_dir, "info"), "r") as f:
            info = json.load(f)

        assert info["format_version"] == "1.0"
        assert info["type"] == "seg_contact"
        assert info["resolution"] == [16, 16, 40]
        assert info["voxel_offset"] == [100, 200, 50]
        assert info["size"] == [1000, 1000, 500]
        assert info["chunk_size"] == [256, 256, 128]
        assert info["max_contact_span"] == 512


def test_from_path_loads_info():
    """Test loading backend from existing info file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # First create and write
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(100, 200, 50),
            size=Vec3D(1000, 1000, 500),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )
        backend.write_info()

        # Then load from path
        loaded = SegContactLayerBackend.from_path(temp_dir)

        assert loaded.resolution == Vec3D(16, 16, 40)
        assert loaded.voxel_offset == Vec3D(100, 200, 50)
        assert loaded.size == Vec3D(1000, 1000, 500)
        assert loaded.chunk_size == Vec3D(256, 256, 128)
        assert loaded.max_contact_span == 512


def test_from_path_missing_info_raises():
    """Test that from_path raises when info file doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            SegContactLayerBackend.from_path(temp_dir)


# --- Chunk write/read tests ---


def make_seg_contact(
    id: int,
    seg_a: int,
    seg_b: int,
    com: tuple[float, float, float],
    n_faces: int = 3,
) -> SegContact:
    """Helper to create a SegContact for testing."""
    contact_faces = np.array(
        [[com[0] + i, com[1] + i, com[2] + i, 0.5 + i * 0.1] for i in range(n_faces)],
        dtype=np.float32,
    )
    return SegContact(
        id=id,
        seg_a=seg_a,
        seg_b=seg_b,
        com=Vec3D(*com),
        contact_faces=contact_faces,
    )


def test_write_chunk_creates_file():
    """Test that write_chunk creates a chunk file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(1000, 1000, 500),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )

        contact = make_seg_contact(id=1, seg_a=100, seg_b=200, com=(100.0, 100.0, 100.0))
        backend.write_chunk((0, 0, 0), [contact])

        chunk_path = backend.get_chunk_path((0, 0, 0))
        assert os.path.exists(chunk_path)


def test_write_read_chunk_single_contact():
    """Test round-trip of a single contact through write_chunk/read_chunk."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(1000, 1000, 500),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )

        contact = make_seg_contact(
            id=42, seg_a=100, seg_b=200, com=(100.0, 200.0, 300.0), n_faces=5
        )
        backend.write_chunk((0, 0, 0), [contact])

        contacts_read = backend.read_chunk((0, 0, 0))

        assert len(contacts_read) == 1
        c = contacts_read[0]
        assert c.id == 42
        assert c.seg_a == 100
        assert c.seg_b == 200
        assert c.com == Vec3D(100.0, 200.0, 300.0)
        assert c.contact_faces.shape == (5, 4)
        np.testing.assert_array_almost_equal(c.contact_faces, contact.contact_faces)


def test_write_read_chunk_multiple_contacts():
    """Test round-trip of multiple contacts in a single chunk."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(1000, 1000, 500),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )

        contacts = [
            make_seg_contact(id=1, seg_a=100, seg_b=200, com=(100.0, 100.0, 100.0), n_faces=3),
            make_seg_contact(id=2, seg_a=100, seg_b=300, com=(200.0, 200.0, 200.0), n_faces=7),
            make_seg_contact(id=3, seg_a=200, seg_b=300, com=(300.0, 300.0, 300.0), n_faces=1),
        ]
        backend.write_chunk((0, 0, 0), contacts)

        contacts_read = backend.read_chunk((0, 0, 0))

        assert len(contacts_read) == 3
        # Verify each contact
        for orig, read in zip(contacts, contacts_read):
            assert read.id == orig.id
            assert read.seg_a == orig.seg_a
            assert read.seg_b == orig.seg_b
            assert read.com == orig.com
            np.testing.assert_array_almost_equal(read.contact_faces, orig.contact_faces)


def test_write_read_chunk_empty():
    """Test writing and reading an empty chunk."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(1000, 1000, 500),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )

        backend.write_chunk((0, 0, 0), [])

        contacts_read = backend.read_chunk((0, 0, 0))
        assert len(contacts_read) == 0


def test_read_chunk_nonexistent_returns_empty():
    """Test reading a chunk that doesn't exist returns empty list."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(1000, 1000, 500),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )

        contacts_read = backend.read_chunk((0, 0, 0))
        assert len(contacts_read) == 0


def test_write_read_chunk_with_partner_metadata():
    """Test round-trip of contact with partner_metadata."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(1000, 1000, 500),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )

        contact = SegContact(
            id=1,
            seg_a=100,
            seg_b=200,
            com=Vec3D(100.0, 100.0, 100.0),
            contact_faces=np.array([[100, 100, 100, 0.5]], dtype=np.float32),
            partner_metadata={100: {"type": "axon"}, 200: {"type": "dendrite"}},
        )
        backend.write_chunk((0, 0, 0), [contact])

        contacts_read = backend.read_chunk((0, 0, 0))

        assert len(contacts_read) == 1
        assert contacts_read[0].partner_metadata == {
            100: {"type": "axon"},
            200: {"type": "dendrite"},
        }


def test_write_read_chunk_with_no_partner_metadata():
    """Test round-trip of contact without partner_metadata."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(1000, 1000, 500),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )

        contact = SegContact(
            id=1,
            seg_a=100,
            seg_b=200,
            com=Vec3D(100.0, 100.0, 100.0),
            contact_faces=np.array([[100, 100, 100, 0.5]], dtype=np.float32),
            partner_metadata=None,
        )
        backend.write_chunk((0, 0, 0), [contact])

        contacts_read = backend.read_chunk((0, 0, 0))

        assert len(contacts_read) == 1
        assert contacts_read[0].partner_metadata is None


def test_write_read_chunk_large_contact_faces():
    """Test contact with many faces."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(1000, 1000, 500),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )

        n_faces = 1000
        contact_faces = np.random.rand(n_faces, 4).astype(np.float32)
        contact = SegContact(
            id=1,
            seg_a=100,
            seg_b=200,
            com=Vec3D(100.0, 100.0, 100.0),
            contact_faces=contact_faces,
        )
        backend.write_chunk((0, 0, 0), [contact])

        contacts_read = backend.read_chunk((0, 0, 0))

        assert len(contacts_read) == 1
        assert contacts_read[0].contact_faces.shape == (n_faces, 4)
        np.testing.assert_array_almost_equal(contacts_read[0].contact_faces, contact_faces)


# --- High-level read/write tests ---


def test_write_distributes_contacts_to_chunks():
    """Test that write() distributes contacts to correct chunks based on COM."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(1000, 1000, 500),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )
        backend.write_info()

        # chunk_size_nm in x = 256 * 16 = 4096
        # Contact 1: COM in chunk (0,0,0)
        # Contact 2: COM in chunk (1,0,0)
        contacts = [
            make_seg_contact(id=1, seg_a=100, seg_b=200, com=(100.0, 100.0, 100.0)),
            make_seg_contact(id=2, seg_a=100, seg_b=300, com=(5000.0, 100.0, 100.0)),  # > 4096
        ]

        idx = VolumetricIndex(
            resolution=Vec3D(16, 16, 40),
            bbox=BBox3D.from_slices((slice(0, 512), slice(0, 256), slice(0, 128))),
        )
        backend.write(idx, contacts)

        # Check chunk (0,0,0)
        chunk_0 = backend.read_chunk((0, 0, 0))
        assert len(chunk_0) == 1
        assert chunk_0[0].id == 1

        # Check chunk (1,0,0)
        chunk_1 = backend.read_chunk((1, 0, 0))
        assert len(chunk_1) == 1
        assert chunk_1[0].id == 2


def test_read_filters_by_bbox():
    """Test that read() filters contacts to those within the query bbox."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(1000, 1000, 500),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )
        backend.write_info()

        # Write contacts at different positions within chunk (0,0,0)
        # All within first chunk (0-4096 nm in x)
        contacts = [
            make_seg_contact(id=1, seg_a=100, seg_b=200, com=(100.0, 100.0, 100.0)),
            make_seg_contact(id=2, seg_a=100, seg_b=300, com=(2000.0, 100.0, 100.0)),
            make_seg_contact(id=3, seg_a=200, seg_b=300, com=(3500.0, 100.0, 100.0)),
        ]
        backend.write_chunk((0, 0, 0), contacts)

        # Query only first half of chunk (0-128 voxels = 0-2048 nm in x)
        idx = VolumetricIndex(
            resolution=Vec3D(16, 16, 40),
            bbox=BBox3D.from_slices((slice(0, 128), slice(0, 256), slice(0, 128))),
        )
        result = backend.read(idx)

        # Should only get contacts 1 and 2 (COM < 2048 nm)
        assert len(result) == 2
        ids = {c.id for c in result}
        assert ids == {1, 2}


def test_read_spans_multiple_chunks():
    """Test that read() can span multiple chunks."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(1000, 1000, 500),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )
        backend.write_info()

        # Write to chunk (0,0,0)
        backend.write_chunk(
            (0, 0, 0),
            [
                make_seg_contact(id=1, seg_a=100, seg_b=200, com=(100.0, 100.0, 100.0)),
            ],
        )

        # Write to chunk (1,0,0)
        backend.write_chunk(
            (1, 0, 0),
            [
                make_seg_contact(id=2, seg_a=100, seg_b=300, com=(5000.0, 100.0, 100.0)),
            ],
        )

        # Query spanning both chunks (0-512 voxels in x)
        idx = VolumetricIndex(
            resolution=Vec3D(16, 16, 40),
            bbox=BBox3D.from_slices((slice(0, 512), slice(0, 256), slice(0, 128))),
        )
        result = backend.read(idx)

        assert len(result) == 2
        ids = {c.id for c in result}
        assert ids == {1, 2}


def test_read_empty_region():
    """Test reading from a region with no contacts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(1000, 1000, 500),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )
        backend.write_info()

        idx = VolumetricIndex(
            resolution=Vec3D(16, 16, 40),
            bbox=BBox3D.from_slices((slice(0, 256), slice(0, 256), slice(0, 128))),
        )
        result = backend.read(idx)

        assert len(result) == 0


def test_round_trip_full():
    """Full round-trip test: write via write(), read via read()."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(1000, 1000, 500),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )
        backend.write_info()

        contacts = [
            make_seg_contact(id=1, seg_a=100, seg_b=200, com=(100.0, 100.0, 100.0), n_faces=5),
            make_seg_contact(id=2, seg_a=100, seg_b=300, com=(200.0, 200.0, 200.0), n_faces=10),
        ]

        write_idx = VolumetricIndex(
            resolution=Vec3D(16, 16, 40),
            bbox=BBox3D.from_slices((slice(0, 256), slice(0, 256), slice(0, 128))),
        )
        backend.write(write_idx, contacts)

        # Read back
        read_idx = VolumetricIndex(
            resolution=Vec3D(16, 16, 40),
            bbox=BBox3D.from_slices((slice(0, 256), slice(0, 256), slice(0, 128))),
        )
        result = backend.read(read_idx)

        assert len(result) == 2
        result_by_id = {c.id: c for c in result}

        for orig in contacts:
            read = result_by_id[orig.id]
            assert read.seg_a == orig.seg_a
            assert read.seg_b == orig.seg_b
            assert read.com == orig.com
            np.testing.assert_array_almost_equal(read.contact_faces, orig.contact_faces)

import tempfile

import numpy as np
import pytest

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric.seg_contact import SegContact, SegContactLayerBackend
from zetta_utils.training.datasets.sample_indexers import (
    SegContactIndexer,
    build_seg_contact_indexer,
)


def make_backend_with_contacts(temp_dir: str) -> SegContactLayerBackend:
    """Create a backend with some test contacts."""
    backend = SegContactLayerBackend(
        path=temp_dir,
        resolution=Vec3D(16, 16, 40),
        voxel_offset=Vec3D(0, 0, 0),
        size=Vec3D(1000, 1000, 500),
        chunk_size=Vec3D(256, 256, 128),
        max_contact_span=512,
    )
    backend.write_info()

    # Create some test contacts in chunk (0,0,0)
    contacts = [
        SegContact(
            id=1,
            seg_a=100,
            seg_b=200,
            com=Vec3D(100.0, 100.0, 100.0),  # nm
            contact_faces=np.array([[1.0, 2.0, 3.0, 10.0]], dtype=np.float32),
        ),
        SegContact(
            id=2,
            seg_a=100,
            seg_b=300,
            com=Vec3D(200.0, 200.0, 200.0),
            contact_faces=np.array([[4.0, 5.0, 6.0, 20.0]], dtype=np.float32),
        ),
        SegContact(
            id=3,
            seg_a=200,
            seg_b=300,
            com=Vec3D(300.0, 300.0, 300.0),
            contact_faces=np.array([[7.0, 8.0, 9.0, 30.0]], dtype=np.float32),
        ),
    ]
    backend._write_contacts_chunk((0, 0, 0), contacts)  # pylint: disable=protected-access

    # Create contacts in chunk (1,0,0)
    contacts2 = [
        SegContact(
            id=4,
            seg_a=400,
            seg_b=500,
            com=Vec3D(5000.0, 100.0, 100.0),  # In chunk (1,0,0) at 16nm resolution
            contact_faces=np.array([[10.0, 11.0, 12.0, 40.0]], dtype=np.float32),
        ),
    ]
    backend._write_contacts_chunk((1, 0, 0), contacts2)  # pylint: disable=protected-access

    return backend


def test_seg_contact_indexer_full_layer():
    """Test indexer covers all contacts in layer."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend_with_contacts(temp_dir)

        indexer = SegContactIndexer(backend=backend)

        assert len(indexer) == 4

        # Check we can access all contacts
        contact_ids = {indexer(i).id for i in range(len(indexer))}
        assert contact_ids == {1, 2, 3, 4}


def test_seg_contact_indexer_with_bbox():
    """Test indexer with restricted bounding box."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend_with_contacts(temp_dir)

        # Only cover chunk (0,0,0)
        bbox = BBox3D.from_coords(
            start_coord=[0, 0, 0],
            end_coord=[256, 256, 128],
            resolution=[16, 16, 40],
        )
        indexer = SegContactIndexer(
            backend=backend,
            bbox=bbox,
            resolution=[16, 16, 40],
        )

        assert len(indexer) == 3

        contact_ids = {indexer(i).id for i in range(len(indexer))}
        assert contact_ids == {1, 2, 3}


def test_seg_contact_indexer_returns_correct_contact():
    """Test that indexer returns contacts with correct data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend_with_contacts(temp_dir)

        indexer = SegContactIndexer(backend=backend)

        # Find contact with id=1
        contact = None
        for i in range(len(indexer)):
            c = indexer(i)
            if c.id == 1:
                contact = c
                break

        assert contact is not None
        assert contact.seg_a == 100
        assert contact.seg_b == 200
        assert contact.com == Vec3D(100.0, 100.0, 100.0)


def test_seg_contact_indexer_index_out_of_range():
    """Test that out of range index raises IndexError."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend_with_contacts(temp_dir)

        indexer = SegContactIndexer(backend=backend)

        with pytest.raises(IndexError):
            indexer(100)

        with pytest.raises(IndexError):
            indexer(-1)


def test_seg_contact_indexer_empty_layer():
    """Test indexer with no contacts."""
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

        indexer = SegContactIndexer(backend=backend)

        assert len(indexer) == 0


def test_seg_contact_indexer_bbox_without_resolution_fails():
    """Test that providing bbox without resolution raises error."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend_with_contacts(temp_dir)

        bbox = BBox3D.from_coords(
            start_coord=[0, 0, 0],
            end_coord=[256, 256, 128],
            resolution=[16, 16, 40],
        )

        with pytest.raises(ValueError, match="resolution is required"):
            SegContactIndexer(backend=backend, bbox=bbox)


def test_build_seg_contact_indexer_function():
    """Test builder function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend_with_contacts(temp_dir)

        indexer = build_seg_contact_indexer(backend=backend)

        assert isinstance(indexer, SegContactIndexer)
        assert len(indexer) == 4

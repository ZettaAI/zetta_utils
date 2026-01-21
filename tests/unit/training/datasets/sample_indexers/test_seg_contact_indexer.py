import tempfile

import numpy as np
import pytest

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.seg_contact import SegContact, SegContactLayerBackend
from zetta_utils.training.datasets.sample_indexers import SegContactIndexer


def make_test_layer(temp_dir: str) -> None:
    """Create a test seg_contact layer with info file."""
    backend = SegContactLayerBackend(
        path=temp_dir,
        resolution=Vec3D(16, 16, 40),
        voxel_offset=Vec3D(0, 0, 0),
        size=Vec3D(512, 512, 256),
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
            com=Vec3D(100.0, 100.0, 100.0),
            contact_faces=np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32),
            representative_points={100: Vec3D(90.0, 90.0, 90.0), 200: Vec3D(110.0, 110.0, 110.0)},
        ),
    ]
    backend.write_chunk((0, 0, 0), contacts)


def test_seg_contact_indexer_len():
    """Test indexer length matches number of chunks."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_test_layer(temp_dir)

        indexer = SegContactIndexer(path=temp_dir)

        # 512/256=2 chunks in x, 512/256=2 in y, 256/128=2 in z => 8 total chunks
        assert len(indexer) == 8


def test_seg_contact_indexer_returns_volumetric_index():
    """Test __call__ returns VolumetricIndex."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_test_layer(temp_dir)

        indexer = SegContactIndexer(path=temp_dir)
        result = indexer(0)

        assert isinstance(result, VolumetricIndex)
        assert result.resolution is not None


def test_seg_contact_indexer_index_out_of_range():
    """Test out of range index raises IndexError."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_test_layer(temp_dir)

        indexer = SegContactIndexer(path=temp_dir)

        with pytest.raises(IndexError):
            indexer(100)

        with pytest.raises(IndexError):
            indexer(-1)


def test_seg_contact_indexer_with_bbox():
    """Test indexer with restricted bounding box."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_test_layer(temp_dir)

        # Only cover first chunk in each dimension
        bbox = BBox3D.from_coords(
            start_coord=[0, 0, 0],
            end_coord=[256, 256, 128],
            resolution=[16, 16, 40],
        )
        indexer = SegContactIndexer(
            path=temp_dir,
            bbox=bbox,
            resolution=[16, 16, 40],
        )

        # Only 1 chunk intersects with bbox
        assert len(indexer) == 1


def test_seg_contact_indexer_bbox_filters_chunks():
    """Test that bbox filters out non-overlapping chunks."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_test_layer(temp_dir)

        # Bbox in second half of x dimension
        bbox = BBox3D.from_coords(
            start_coord=[256, 0, 0],
            end_coord=[512, 256, 128],
            resolution=[16, 16, 40],
        )
        indexer = SegContactIndexer(
            path=temp_dir,
            bbox=bbox,
            resolution=[16, 16, 40],
        )

        # 1 chunk in x (second half), 1 in y, 1 in z
        assert len(indexer) == 1


def test_seg_contact_indexer_deterministic_order():
    """Test that chunk order is deterministic."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_test_layer(temp_dir)

        indexer1 = SegContactIndexer(path=temp_dir)
        indexer2 = SegContactIndexer(path=temp_dir)

        for i in range(len(indexer1)):
            idx1 = indexer1(i)
            idx2 = indexer2(i)
            assert idx1.bbox == idx2.bbox


def test_seg_contact_indexer_chunk_bbox_correct():
    """Test that returned VolumetricIndex has correct bbox for chunk."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_test_layer(temp_dir)

        indexer = SegContactIndexer(path=temp_dir)

        # First chunk should cover (0,0,0) to (256,256,128) voxels
        # In nm at resolution (16,16,40): (0,0,0) to (4096,4096,5120)
        result = indexer(0)

        # Check bbox start
        assert result.bbox.start[0] == 0
        assert result.bbox.start[1] == 0
        assert result.bbox.start[2] == 0


def test_seg_contact_indexer_uses_layer_resolution():
    """Test that indexer uses layer resolution from info file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_test_layer(temp_dir)

        indexer = SegContactIndexer(path=temp_dir)
        result = indexer(0)

        # Should use resolution from info file
        assert result.resolution == Vec3D(16, 16, 40)


def test_seg_contact_indexer_custom_resolution():
    """Test that custom resolution overrides layer resolution."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_test_layer(temp_dir)

        indexer = SegContactIndexer(path=temp_dir, resolution=[8, 8, 20])
        result = indexer(0)

        assert result.resolution == Vec3D(8, 8, 20)


def test_seg_contact_indexer_bbox_intersection():
    """Test that returned bbox is intersection with filter bbox."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_test_layer(temp_dir)

        # Bbox that partially overlaps first chunk
        # First chunk is (0,0,0) to (4096,4096,5120) nm
        bbox = BBox3D.from_coords(
            start_coord=[128, 128, 64],  # in voxels at res (16,16,40)
            end_coord=[384, 384, 192],
            resolution=[16, 16, 40],
        )
        indexer = SegContactIndexer(
            path=temp_dir,
            bbox=bbox,
            resolution=[16, 16, 40],
        )

        result = indexer(0)

        # Result should be intersection of chunk bbox and filter bbox
        # Chunk 0 is (0,0,0) to (256,256,128) voxels
        # Filter is (128,128,64) to (384,384,192) voxels
        # Intersection is (128,128,64) to (256,256,128) voxels
        expected_start_nm = Vec3D(128 * 16, 128 * 16, 64 * 40)
        expected_end_nm = Vec3D(256 * 16, 256 * 16, 128 * 40)

        assert result.bbox.start == expected_start_nm
        assert result.bbox.end == expected_end_nm

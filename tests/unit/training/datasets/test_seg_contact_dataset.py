import json
import tempfile

import numpy as np
import torch

from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric.seg_contact import (
    SegContact,
    SegContactLayerBackend,
    VolumetricSegContactLayer,
)
from zetta_utils.training.datasets import SegContactDataset
from zetta_utils.training.datasets.sample_indexers import SegContactIndexer


def make_layer_with_contacts(temp_dir: str) -> VolumetricSegContactLayer:
    """Create a layer with test contacts including pointclouds and merge decisions."""
    backend = SegContactLayerBackend(
        path=temp_dir,
        resolution=Vec3D(16, 16, 40),
        voxel_offset=Vec3D(0, 0, 0),
        size=Vec3D(512, 512, 256),
        chunk_size=Vec3D(256, 256, 128),
        max_contact_span=512,
    )

    # Write info with pointcloud configs and merge decision authorities
    info = {
        "format_version": "1.0",
        "type": "seg_contact",
        "resolution": list(backend.resolution),
        "voxel_offset": list(backend.voxel_offset),
        "size": list(backend.size),
        "chunk_size": list(backend.chunk_size),
        "max_contact_span": backend.max_contact_span,
        "local_point_clouds": [{"radius_nm": 500, "n_points": 64}],
        "merge_decisions": ["human"],
    }
    with open(f"{temp_dir}/info", "w", encoding="utf-8") as f:
        json.dump(info, f)

    contacts = [
        SegContact(
            id=1,
            seg_a=100,
            seg_b=200,
            com=Vec3D(100.0, 100.0, 100.0),
            contact_faces=np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32),
            representative_points={100: Vec3D(90.0, 90.0, 90.0), 200: Vec3D(110.0, 110.0, 110.0)},
            local_pointclouds={
                (500, 64): {
                    100: np.random.randn(64, 3).astype(np.float32),
                    200: np.random.randn(64, 3).astype(np.float32),
                }
            },
            merge_decisions={"human": True},
        ),
        SegContact(
            id=2,
            seg_a=100,
            seg_b=300,
            com=Vec3D(200.0, 200.0, 200.0),
            contact_faces=np.array([[4.0, 5.0, 6.0, 0.8]], dtype=np.float32),
            representative_points={
                100: Vec3D(190.0, 190.0, 190.0),
                300: Vec3D(210.0, 210.0, 210.0),
            },
            local_pointclouds={
                (500, 64): {
                    100: np.random.randn(64, 3).astype(np.float32),
                    300: np.random.randn(64, 3).astype(np.float32),
                }
            },
            merge_decisions={"human": False},
        ),
    ]
    backend.write_chunk((0, 0, 0), contacts)

    return VolumetricSegContactLayer(backend=backend)


def test_seg_contact_dataset_len():
    """Test dataset length matches number of chunks."""
    with tempfile.TemporaryDirectory() as temp_dir:
        layer = make_layer_with_contacts(temp_dir)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(layer=layer, sample_indexer=indexer)

        # 8 chunks total
        assert len(dataset) == 8


def test_seg_contact_dataset_getitem_basic():
    """Test basic __getitem__ returns expected keys."""
    with tempfile.TemporaryDirectory() as temp_dir:
        layer = make_layer_with_contacts(temp_dir)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(layer=layer, sample_indexer=indexer)

        sample = dataset[0]

        assert "contact_id" in sample
        assert "seg_a" in sample
        assert "seg_b" in sample
        assert "com" in sample
        assert "contact_faces" in sample

        assert isinstance(sample["contact_id"], torch.Tensor)
        assert sample["contact_id"].dtype == torch.int64
        assert sample["seg_a"].dtype == torch.int64
        assert sample["com"].dtype == torch.float32
        # Batch of 2 contacts
        assert sample["com"].shape == (2, 3)


def test_seg_contact_dataset_with_pointclouds():
    """Test dataset returns combined pointcloud when contacts have pointclouds."""
    with tempfile.TemporaryDirectory() as temp_dir:
        layer = make_layer_with_contacts(temp_dir)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(layer=layer, sample_indexer=indexer)

        sample = dataset[0]

        # Pointclouds should be combined into single tensor with segment labels
        assert "pointcloud" in sample
        # Each contact has 64 + 64 = 128 points with 4 channels (x,y,z,label)
        assert sample["pointcloud"].shape == (2, 128, 4)
        assert sample["pointcloud"].dtype == torch.float32


def test_seg_contact_dataset_with_merge_decision():
    """Test dataset returns target when authority specified."""
    with tempfile.TemporaryDirectory() as temp_dir:
        layer = make_layer_with_contacts(temp_dir)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(
            layer=layer, sample_indexer=indexer, merge_decision_authority="human"
        )

        sample = dataset[0]

        assert "target" in sample
        assert sample["target"].dtype == torch.float32
        # One should be 1.0 (True), one should be 0.0 (False)
        targets = set(sample["target"].tolist())
        assert targets == {0.0, 1.0}


def test_seg_contact_dataset_no_target_without_authority():
    """Test target not returned when authority not specified."""
    with tempfile.TemporaryDirectory() as temp_dir:
        layer = make_layer_with_contacts(temp_dir)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(layer=layer, sample_indexer=indexer)

        sample = dataset[0]

        assert "target" not in sample


def test_seg_contact_dataset_contact_faces_shape():
    """Test contact_faces has correct shape with padding."""
    with tempfile.TemporaryDirectory() as temp_dir:
        layer = make_layer_with_contacts(temp_dir)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(layer=layer, sample_indexer=indexer, max_contact_faces=100)

        sample = dataset[0]

        # 2 contacts, each padded to 100 faces, 4 channels (x,y,z,affinity)
        assert sample["contact_faces"].shape == (2, 100, 4)


def test_seg_contact_dataset_empty_chunk_returns_empty():
    """Test dataset returns empty tensors for chunk with no contacts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(512, 512, 256),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )
        backend.write_info()

        layer = VolumetricSegContactLayer(backend=backend)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(layer=layer, sample_indexer=indexer)

        sample = dataset[0]

        # Empty chunk should return empty tensors
        assert sample["contact_id"].shape[0] == 0
        assert sample["seg_a"].shape[0] == 0


def test_seg_contact_dataset_include_contact_faces_in_pointcloud():
    """Test contact_faces can be included in pointcloud."""
    with tempfile.TemporaryDirectory() as temp_dir:
        layer = make_layer_with_contacts(temp_dir)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(
            layer=layer,
            sample_indexer=indexer,
            max_contact_faces=10,
            include_contact_faces_in_pointcloud=True,
        )

        sample = dataset[0]

        # Pointcloud should include contact_faces
        # 64 + 64 (from both segments) + 10 (contact faces) = 138 points
        assert sample["pointcloud"].shape == (2, 138, 4)


def test_seg_contact_dataset_truncates_large_contact_faces():
    """Test that contact_faces larger than max_contact_faces is truncated."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(512, 512, 256),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )

        info = {
            "format_version": "1.0",
            "type": "seg_contact",
            "resolution": list(backend.resolution),
            "voxel_offset": list(backend.voxel_offset),
            "size": list(backend.size),
            "chunk_size": list(backend.chunk_size),
            "max_contact_span": backend.max_contact_span,
            "local_point_clouds": [{"radius_nm": 500, "n_points": 64}],
        }
        with open(f"{temp_dir}/info", "w", encoding="utf-8") as f:
            json.dump(info, f)

        # Create contact with many faces
        large_faces = np.random.randn(100, 4).astype(np.float32)
        contacts = [
            SegContact(
                id=1,
                seg_a=100,
                seg_b=200,
                com=Vec3D(100.0, 100.0, 100.0),
                contact_faces=large_faces,
                representative_points={
                    100: Vec3D(90.0, 90.0, 90.0),
                    200: Vec3D(110.0, 110.0, 110.0),
                },
                local_pointclouds={
                    (500, 64): {
                        100: np.random.randn(64, 3).astype(np.float32),
                        200: np.random.randn(64, 3).astype(np.float32),
                    }
                },
            ),
        ]
        backend.write_chunk((0, 0, 0), contacts)

        layer = VolumetricSegContactLayer(backend=backend)
        indexer = SegContactIndexer(path=temp_dir)
        # Use small max_contact_faces to test truncation
        dataset = SegContactDataset(layer=layer, sample_indexer=indexer, max_contact_faces=50)
        sample = dataset[0]

        # Should be truncated to 50
        assert sample["contact_faces"].shape == (1, 50, 4)


def test_seg_contact_dataset_contact_faces_exact_size():
    """Test that contact_faces with exact size is returned unchanged."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(512, 512, 256),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )

        info = {
            "format_version": "1.0",
            "type": "seg_contact",
            "resolution": list(backend.resolution),
            "voxel_offset": list(backend.voxel_offset),
            "size": list(backend.size),
            "chunk_size": list(backend.chunk_size),
            "max_contact_span": backend.max_contact_span,
            "local_point_clouds": [{"radius_nm": 500, "n_points": 64}],
        }
        with open(f"{temp_dir}/info", "w", encoding="utf-8") as f:
            json.dump(info, f)

        # Create contact with exactly 10 faces
        exact_faces = np.random.randn(10, 4).astype(np.float32)
        contacts = [
            SegContact(
                id=1,
                seg_a=100,
                seg_b=200,
                com=Vec3D(100.0, 100.0, 100.0),
                contact_faces=exact_faces,
                representative_points={
                    100: Vec3D(90.0, 90.0, 90.0),
                    200: Vec3D(110.0, 110.0, 110.0),
                },
                local_pointclouds={
                    (500, 64): {
                        100: np.random.randn(64, 3).astype(np.float32),
                        200: np.random.randn(64, 3).astype(np.float32),
                    }
                },
            ),
        ]
        backend.write_chunk((0, 0, 0), contacts)

        layer = VolumetricSegContactLayer(backend=backend)
        indexer = SegContactIndexer(path=temp_dir)
        # Use max_contact_faces=10 to test exact match case
        dataset = SegContactDataset(layer=layer, sample_indexer=indexer, max_contact_faces=10)
        sample = dataset[0]

        # Should have exactly 10 faces (no padding or truncation)
        assert sample["contact_faces"].shape == (1, 10, 4)


def test_seg_contact_dataset_skips_contacts_missing_pointcloud():
    """Test that contacts missing required pointcloud config are skipped."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(512, 512, 256),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )

        # Info specifies pointcloud config, but some contacts won't have it
        info = {
            "format_version": "1.0",
            "type": "seg_contact",
            "resolution": list(backend.resolution),
            "voxel_offset": list(backend.voxel_offset),
            "size": list(backend.size),
            "chunk_size": list(backend.chunk_size),
            "max_contact_span": backend.max_contact_span,
            "local_point_clouds": [{"radius_nm": 500, "n_points": 64}],
        }
        with open(f"{temp_dir}/info", "w", encoding="utf-8") as f:
            json.dump(info, f)

        contacts = [
            # Contact with pointcloud - should be included
            SegContact(
                id=1,
                seg_a=100,
                seg_b=200,
                com=Vec3D(100.0, 100.0, 100.0),
                contact_faces=np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32),
                representative_points={
                    100: Vec3D(90.0, 90.0, 90.0),
                    200: Vec3D(110.0, 110.0, 110.0),
                },
                local_pointclouds={
                    (500, 64): {
                        100: np.random.randn(64, 3).astype(np.float32),
                        200: np.random.randn(64, 3).astype(np.float32),
                    }
                },
            ),
            # Contact without pointcloud - should be skipped
            SegContact(
                id=2,
                seg_a=300,
                seg_b=400,
                com=Vec3D(120.0, 120.0, 100.0),
                contact_faces=np.array([[4.0, 5.0, 6.0, 0.8]], dtype=np.float32),
                representative_points={
                    300: Vec3D(110.0, 110.0, 90.0),
                    400: Vec3D(130.0, 130.0, 110.0),
                },
                local_pointclouds=None,
            ),
        ]
        backend.write_chunk((0, 0, 0), contacts)

        layer = VolumetricSegContactLayer(backend=backend)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(layer=layer, sample_indexer=indexer)

        sample = dataset[0]

        # Only 1 contact should be in sample (the one with pointcloud)
        assert sample["contact_id"].shape[0] == 1
        assert sample["contact_id"][0].item() == 1


def test_seg_contact_dataset_skips_contacts_missing_segment_pointcloud():
    """Test that contacts missing one segment's pointcloud are skipped."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = SegContactLayerBackend(
            path=temp_dir,
            resolution=Vec3D(16, 16, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(512, 512, 256),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
        )

        info = {
            "format_version": "1.0",
            "type": "seg_contact",
            "resolution": list(backend.resolution),
            "voxel_offset": list(backend.voxel_offset),
            "size": list(backend.size),
            "chunk_size": list(backend.chunk_size),
            "max_contact_span": backend.max_contact_span,
            "local_point_clouds": [{"radius_nm": 500, "n_points": 64}],
        }
        with open(f"{temp_dir}/info", "w", encoding="utf-8") as f:
            json.dump(info, f)

        contacts = [
            # Contact with only one segment's pointcloud - should be skipped
            SegContact(
                id=1,
                seg_a=100,
                seg_b=200,
                com=Vec3D(100.0, 100.0, 100.0),
                contact_faces=np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32),
                representative_points={
                    100: Vec3D(90.0, 90.0, 90.0),
                    200: Vec3D(110.0, 110.0, 110.0),
                },
                local_pointclouds={
                    (500, 64): {
                        100: np.random.randn(64, 3).astype(np.float32),
                        # Missing seg_b (200) pointcloud
                    }
                },
            ),
            # Contact with both segments' pointclouds - should be included
            SegContact(
                id=2,
                seg_a=300,
                seg_b=400,
                com=Vec3D(120.0, 120.0, 100.0),
                contact_faces=np.array([[4.0, 5.0, 6.0, 0.8]], dtype=np.float32),
                representative_points={
                    300: Vec3D(110.0, 110.0, 90.0),
                    400: Vec3D(130.0, 130.0, 110.0),
                },
                local_pointclouds={
                    (500, 64): {
                        300: np.random.randn(64, 3).astype(np.float32),
                        400: np.random.randn(64, 3).astype(np.float32),
                    }
                },
            ),
        ]
        backend.write_chunk((0, 0, 0), contacts)

        layer = VolumetricSegContactLayer(backend=backend)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(layer=layer, sample_indexer=indexer)

        sample = dataset[0]

        # Only 1 contact should be in sample (the complete one)
        assert sample["contact_id"].shape[0] == 1
        assert sample["contact_id"][0].item() == 2

import json
import tempfile

import numpy as np
import torch
from cloudfiles import CloudFile

from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric.seg_contact import SegContact, SegContactLayerBackend
from zetta_utils.training.datasets import SegContactDataset


def make_backend_with_contacts(temp_dir: str) -> SegContactLayerBackend:
    """Create a backend with test contacts including pointclouds and merge decisions."""
    backend = SegContactLayerBackend(
        path=temp_dir,
        resolution=Vec3D(16, 16, 40),
        voxel_offset=Vec3D(0, 0, 0),
        size=Vec3D(1000, 1000, 500),
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
    CloudFile(f"{temp_dir}/info").put(json.dumps(info, indent=2).encode("utf-8"))

    contacts = [
        SegContact(
            id=1,
            seg_a=100,
            seg_b=200,
            com=Vec3D(100.0, 100.0, 100.0),
            contact_faces=np.array([[1.0, 2.0, 3.0, 10.0]], dtype=np.float32),
            local_pointclouds={
                "r500_n64": {
                    100: np.random.randn(64, 3).astype(np.float32),
                    200: np.random.randn(64, 3).astype(np.float32),
                }
            },
            merge_decisions={"human": True},
            partner_metadata={100: 1, 200: 2},
        ),
        SegContact(
            id=2,
            seg_a=100,
            seg_b=300,
            com=Vec3D(200.0, 200.0, 200.0),
            contact_faces=np.array([[4.0, 5.0, 6.0, 20.0]], dtype=np.float32),
            local_pointclouds={
                "r500_n64": {
                    100: np.random.randn(64, 3).astype(np.float32),
                    300: np.random.randn(64, 3).astype(np.float32),
                }
            },
            merge_decisions={"human": False},
            partner_metadata={100: 1, 300: 3},
        ),
    ]
    # Use write_chunk to write all data (contacts, pointclouds, merge_decisions)
    backend.write_chunk((0, 0, 0), contacts)

    return backend


def test_seg_contact_dataset_len():
    """Test dataset length matches number of contacts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend_with_contacts(temp_dir)
        dataset = SegContactDataset(backend=backend)

        assert len(dataset) == 2


def test_seg_contact_dataset_getitem_basic():
    """Test basic __getitem__ returns expected keys."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend_with_contacts(temp_dir)
        dataset = SegContactDataset(backend=backend)

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
        assert sample["com"].shape == (3,)


def test_seg_contact_dataset_with_pointclouds():
    """Test dataset returns pointclouds when config specified."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend_with_contacts(temp_dir)
        dataset = SegContactDataset(backend=backend, pointcloud_config="r500_n64")

        sample = dataset[0]

        assert "pointcloud_a" in sample
        assert "pointcloud_b" in sample
        assert sample["pointcloud_a"].shape == (64, 3)
        assert sample["pointcloud_b"].shape == (64, 3)
        assert sample["pointcloud_a"].dtype == torch.float32


def test_seg_contact_dataset_with_merge_decision():
    """Test dataset returns target when authority specified."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend_with_contacts(temp_dir)
        dataset = SegContactDataset(backend=backend, merge_decision_authority="human")

        sample0 = dataset[0]
        sample1 = dataset[1]

        assert "target" in sample0
        assert "target" in sample1
        assert sample0["target"].dtype == torch.float32
        # One should be 1.0 (True), one should be 0.0 (False)
        targets = {sample0["target"].item(), sample1["target"].item()}
        assert targets == {0.0, 1.0}


def test_seg_contact_dataset_with_metadata():
    """Test dataset returns partner metadata."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend_with_contacts(temp_dir)
        dataset = SegContactDataset(backend=backend)

        sample = dataset[0]

        assert "metadata_a" in sample
        assert "metadata_b" in sample
        # Metadata is passed through as-is (can be any type)
        assert sample["metadata_a"] in [1, 2, 3]
        assert sample["metadata_b"] in [1, 2, 3]


def test_seg_contact_dataset_no_pointclouds_without_config():
    """Test pointclouds not returned when config not specified."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend_with_contacts(temp_dir)
        dataset = SegContactDataset(backend=backend)

        sample = dataset[0]

        assert "pointcloud_a" not in sample
        assert "pointcloud_b" not in sample


def test_seg_contact_dataset_no_target_without_authority():
    """Test target not returned when authority not specified."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend_with_contacts(temp_dir)
        dataset = SegContactDataset(backend=backend)

        sample = dataset[0]

        assert "target" not in sample


def test_seg_contact_dataset_full_config():
    """Test dataset with all options enabled."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend_with_contacts(temp_dir)
        dataset = SegContactDataset(
            backend=backend,
            pointcloud_config="r500_n64",
            merge_decision_authority="human",
        )

        sample = dataset[0]

        # All keys should be present
        expected_keys = {
            "contact_id",
            "seg_a",
            "seg_b",
            "com",
            "contact_faces",
            "pointcloud_a",
            "pointcloud_b",
            "target",
            "metadata_a",
            "metadata_b",
        }
        assert set(sample.keys()) == expected_keys

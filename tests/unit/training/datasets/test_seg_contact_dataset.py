import json
import tempfile

import numpy as np
import pytest
import torch

from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric.seg_contact import (
    SegContact,
    SegContactLayerBackend,
    VolumetricSegContactLayer,
)
from zetta_utils.training.datasets import SegContactDataset
from zetta_utils.training.datasets.sample_indexers import SegContactIndexer
from zetta_utils.training.datasets.seg_contact_dataset import (
    _apply_channel_mask,
    _broadcast_to_list,
    _pad_or_truncate,
)


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
        dataset = SegContactDataset(
            layer=layer, sample_indexer=indexer, merge_decision_authority="human"
        )

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
        dataset = SegContactDataset(
            layer=layer, sample_indexer=indexer, merge_decision_authority="human"
        )

        sample = dataset[0]

        # Pointclouds should be combined into single tensor with segment labels
        assert "pointcloud" in sample
        # Each contact has 64 + 64 = 128 points with 4 channels (x,y,z,label)
        assert sample["pointcloud"].shape == (2, 128, 4)
        assert sample["pointcloud"].dtype == torch.float32


def test_seg_contact_dataset_with_merge_decision():
    """Test dataset returns merge target when authority specified."""
    with tempfile.TemporaryDirectory() as temp_dir:
        layer = make_layer_with_contacts(temp_dir)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(
            layer=layer, sample_indexer=indexer, merge_decision_authority="human"
        )

        sample = dataset[0]

        assert "merge" in sample
        assert sample["merge"].dtype == torch.float32
        # Shape is [B, 1]
        assert sample["merge"].shape == (2, 1)
        # One should be 1.0 (True), one should be 0.0 (False)
        targets = set(sample["merge"].squeeze().tolist())
        assert targets == {0.0, 1.0}


def test_seg_contact_dataset_no_merge_without_authority():
    """Test that without authority, dataset returns empty (no targets means skip)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        layer = make_layer_with_contacts(temp_dir)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(layer=layer, sample_indexer=indexer)

        sample = dataset[0]

        # Without merge_decision_authority, contacts with pointclouds but no
        # targets return empty dict (so RebatchingDataLoader skips them)
        assert sample == {} or "merge" not in sample


def test_seg_contact_dataset_contact_faces_shape():
    """Test contact_faces has correct shape with padding."""
    with tempfile.TemporaryDirectory() as temp_dir:
        layer = make_layer_with_contacts(temp_dir)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(
            layer=layer, sample_indexer=indexer, merge_decision_authority="human",
            max_contact_faces=100,
        )

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
            merge_decision_authority="human",
            max_contact_faces=10,
            include_contact_faces_in_pointcloud=True,
        )

        sample = dataset[0]

        # Pointcloud should include contact_faces
        # 64 + 64 (from both segments) + 1 (each contact has 1 face) = 129 points
        assert sample["pointcloud"].shape == (2, 129, 4)


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
            "merge_decisions": ["human"],
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
                merge_decisions={"human": True},
            ),
        ]
        backend.write_chunk((0, 0, 0), contacts)

        layer = VolumetricSegContactLayer(backend=backend)
        indexer = SegContactIndexer(path=temp_dir)
        # Use small max_contact_faces to test truncation
        dataset = SegContactDataset(
            layer=layer, sample_indexer=indexer, merge_decision_authority="human",
            max_contact_faces=50,
        )
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
            "merge_decisions": ["human"],
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
                merge_decisions={"human": True},
            ),
        ]
        backend.write_chunk((0, 0, 0), contacts)

        layer = VolumetricSegContactLayer(backend=backend)
        indexer = SegContactIndexer(path=temp_dir)
        # Use max_contact_faces=10 to test exact match case
        dataset = SegContactDataset(
            layer=layer, sample_indexer=indexer, merge_decision_authority="human",
            max_contact_faces=10,
        )
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
            "merge_decisions": ["human"],
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
                merge_decisions={"human": True},
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
                merge_decisions={"human": False},
            ),
        ]
        backend.write_chunk((0, 0, 0), contacts)

        layer = VolumetricSegContactLayer(backend=backend)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(
            layer=layer, sample_indexer=indexer, merge_decision_authority="human"
        )

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
            "merge_decisions": ["human"],
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
                merge_decisions={"human": True},
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
                merge_decisions={"human": False},
            ),
        ]
        backend.write_chunk((0, 0, 0), contacts)

        layer = VolumetricSegContactLayer(backend=backend)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(
            layer=layer, sample_indexer=indexer, merge_decision_authority="human"
        )

        sample = dataset[0]

        # Only 1 contact should be in sample (the complete one)
        assert sample["contact_id"].shape[0] == 1
        assert sample["contact_id"][0].item() == 2


# --- _pad_or_truncate ---


def test_pad_or_truncate_pad():
    t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    result = _pad_or_truncate(t, 4)
    assert result.shape == (4, 2)
    torch.testing.assert_close(result[:2], t)
    assert (result[2:] == 0).all()


def test_pad_or_truncate_truncate():
    t = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
    result = _pad_or_truncate(t, 2)
    assert result.shape == (2, 2)
    torch.testing.assert_close(result, t[:2])


def test_pad_or_truncate_exact():
    t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    result = _pad_or_truncate(t, 2)
    assert result is t  # no-op, same object


# --- _broadcast_to_list ---


def test_broadcast_to_list_scalar():
    assert _broadcast_to_list(0.5, 3) == [0.5, 0.5, 0.5]


def test_broadcast_to_list_string():
    assert _broadcast_to_list("global", 2) == ["global", "global"]


def test_broadcast_to_list_already_list():
    assert _broadcast_to_list([0.1, 0.2], 2) == [0.1, 0.2]


# --- _apply_channel_mask ---


def test_apply_channel_mask_global():
    """Global mask should zero entire channel."""
    torch.manual_seed(0)
    pc = torch.ones((10, 5), dtype=torch.float32)  # xyz + seg_label + affinity

    result = _apply_channel_mask(
        pc,
        mask_probs=1.0,  # always mask
        mask_modes="global",
        mask_global_probs=0.5,
        mask_value=0.0,
    )

    # xyz should be unchanged
    torch.testing.assert_close(result[:, :3], pc[:, :3])
    # At least one feature channel should be zeroed (with prob=1.0)
    assert (result[:, 3] == 0).all() or (result[:, 4] == 0).all()


def test_apply_channel_mask_no_mask_when_prob_zero():
    pc = torch.ones((10, 5), dtype=torch.float32)
    result = _apply_channel_mask(
        pc,
        mask_probs=0.0,
        mask_modes="global",
        mask_global_probs=0.5,
        mask_value=0.0,
    )
    torch.testing.assert_close(result, pc)


def test_apply_channel_mask_xyz_untouched():
    """XYZ channels (0-2) should never be modified."""
    torch.manual_seed(42)
    pc = torch.randn((20, 5), dtype=torch.float32)
    original_xyz = pc[:, :3].clone()

    result = _apply_channel_mask(
        pc,
        mask_probs=1.0,
        mask_modes="global",
        mask_global_probs=1.0,
        mask_value=0.0,
    )

    torch.testing.assert_close(result[:, :3], original_xyz)


def test_apply_channel_mask_per_channel_config():
    """Test spec-like config: segment_label=global, affinity=random."""
    torch.manual_seed(42)
    pc = torch.ones((10, 5), dtype=torch.float32)

    result = _apply_channel_mask(
        pc,
        mask_probs=[1.0, 1.0],  # always mask both
        mask_modes=["global", "random"],
        mask_global_probs=[0.5, 0.2],
        mask_value=0.0,
    )

    # Channel 3 (seg label) should be fully zeroed (global mode, prob=1.0)
    assert (result[:, 3] == 0).all()


def test_apply_channel_mask_partial_skip():
    """With prob < 1.0, some channels are randomly skipped."""
    torch.manual_seed(1)  # seed 1: channel 0 skipped (rand=0.76 >= 0.5), channel 1 masked
    pc = torch.ones((10, 5), dtype=torch.float32)

    result = _apply_channel_mask(
        pc,
        mask_probs=0.5,
        mask_modes="global",
        mask_global_probs=0.5,
        mask_value=0.0,
    )

    # Channel 3 (feat 0) should be UNchanged (skipped)
    assert (result[:, 3] == 1.0).all()
    # Channel 4 (feat 1) should be masked
    assert (result[:, 4] == 0.0).all()


def test_apply_channel_mask_only_3_channels_noop():
    """No feature channels means no masking."""
    pc = torch.ones((10, 3), dtype=torch.float32)
    result = _apply_channel_mask(pc, mask_probs=1.0, mask_modes="global", mask_global_probs=0.5, mask_value=0.0)
    torch.testing.assert_close(result, pc)


# --- affinity filtering ---


def test_seg_contact_dataset_filters_by_affinity():
    """Test that contacts outside [min, max] mean affinity are filtered."""
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
            "merge_decisions": ["human"],
        }
        with open(f"{temp_dir}/info", "w", encoding="utf-8") as f:
            json.dump(info, f)

        contacts = [
            # Low affinity (0.05) - should be filtered out
            SegContact(
                id=1,
                seg_a=100,
                seg_b=200,
                com=Vec3D(100.0, 100.0, 100.0),
                contact_faces=np.array([[1, 2, 3, 0.05]], dtype=np.float32),
                representative_points={100: Vec3D(90.0, 90.0, 90.0), 200: Vec3D(110.0, 110.0, 110.0)},
                local_pointclouds={(500, 64): {
                    100: np.random.randn(64, 3).astype(np.float32),
                    200: np.random.randn(64, 3).astype(np.float32),
                }},
                merge_decisions={"human": True},
            ),
            # In-range affinity (0.25) - should pass
            SegContact(
                id=2,
                seg_a=300,
                seg_b=400,
                com=Vec3D(120.0, 120.0, 100.0),
                contact_faces=np.array([[4, 5, 6, 0.25]], dtype=np.float32),
                representative_points={300: Vec3D(110.0, 110.0, 90.0), 400: Vec3D(130.0, 130.0, 110.0)},
                local_pointclouds={(500, 64): {
                    300: np.random.randn(64, 3).astype(np.float32),
                    400: np.random.randn(64, 3).astype(np.float32),
                }},
                merge_decisions={"human": False},
            ),
            # High affinity (0.9) - should be filtered out
            SegContact(
                id=3,
                seg_a=500,
                seg_b=600,
                com=Vec3D(150.0, 150.0, 100.0),
                contact_faces=np.array([[7, 8, 9, 0.9]], dtype=np.float32),
                representative_points={500: Vec3D(140.0, 140.0, 90.0), 600: Vec3D(160.0, 160.0, 110.0)},
                local_pointclouds={(500, 64): {
                    500: np.random.randn(64, 3).astype(np.float32),
                    600: np.random.randn(64, 3).astype(np.float32),
                }},
                merge_decisions={"human": True},
            ),
        ]
        backend.write_chunk((0, 0, 0), contacts)

        layer = VolumetricSegContactLayer(backend=backend)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(
            layer=layer,
            sample_indexer=indexer,
            merge_decision_authority="human",
            min_mean_affinity=0.1,
            max_mean_affinity=0.4,
        )

        sample = dataset[0]

        # Only contact 2 should pass the affinity filter
        assert sample["contact_id"].shape[0] == 1
        assert sample["contact_id"][0].item() == 2


# --- affinity_channel_mode ---


def test_seg_contact_dataset_per_point_affinity_channel():
    """Test that affinity_channel_mode='per_point' produces 5-channel output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        layer = make_layer_with_contacts(temp_dir)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(
            layer=layer,
            sample_indexer=indexer,
            merge_decision_authority="human",
            include_contact_faces_in_pointcloud=True,
            contact_label=0.0,
            affinity_channel_mode="per_point",
        )

        sample = dataset[0]

        assert "pointcloud" in sample
        # 5 channels: xyz + segment_label + affinity
        assert sample["pointcloud"].shape[2] == 5
        # Segment points should have 0 in affinity channel
        # Contact face points should have non-zero affinity


def test_seg_contact_dataset_mean_affinity_channel():
    """Test affinity_channel_mode='mean' broadcasts mean to all CF points."""
    with tempfile.TemporaryDirectory() as temp_dir:
        layer = make_layer_with_contacts(temp_dir)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(
            layer=layer,
            sample_indexer=indexer,
            merge_decision_authority="human",
            include_contact_faces_in_pointcloud=True,
            contact_label=0.0,
            affinity_channel_mode="mean",
        )

        sample = dataset[0]

        assert "pointcloud" in sample
        assert sample["pointcloud"].shape[2] == 5


def test_seg_contact_dataset_contact_label_none():
    """Test contact_label=None uses affinity as 4th channel for CF points."""
    with tempfile.TemporaryDirectory() as temp_dir:
        layer = make_layer_with_contacts(temp_dir)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(
            layer=layer,
            sample_indexer=indexer,
            merge_decision_authority="human",
            include_contact_faces_in_pointcloud=True,
            contact_label=None,
            affinity_channel_mode="per_point",
        )

        sample = dataset[0]

        assert "pointcloud" in sample
        assert sample["pointcloud"].shape[2] == 5


def test_seg_contact_dataset_contact_faces_original_nm():
    """Test contact_faces_original_nm handling via normalize_pointclouds read_proc."""
    from zetta_utils.layer.volumetric.seg_contact import normalize_pointclouds

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
            "merge_decisions": ["human"],
        }
        with open(f"{temp_dir}/info", "w", encoding="utf-8") as f:
            json.dump(info, f)

        original_faces = np.array([[100, 200, 300, 0.5]], dtype=np.float32)
        contacts = [
            SegContact(
                id=1,
                seg_a=100,
                seg_b=200,
                com=Vec3D(100.0, 100.0, 100.0),
                contact_faces=original_faces.copy(),
                representative_points={100: Vec3D(90.0, 90.0, 90.0), 200: Vec3D(110.0, 110.0, 110.0)},
                local_pointclouds={(500, 64): {
                    100: np.random.randn(64, 3).astype(np.float32),
                    200: np.random.randn(64, 3).astype(np.float32),
                }},
                merge_decisions={"human": True},
            ),
        ]
        backend.write_chunk((0, 0, 0), contacts)

        # Use normalize_pointclouds as a read_proc — this sets contact_faces_original_nm
        from functools import partial
        layer = VolumetricSegContactLayer(
            backend=backend,
            read_procs=[partial(normalize_pointclouds, use_pointcloud_radius=True)],
        )
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(
            layer=layer, sample_indexer=indexer, merge_decision_authority="human",
        )

        sample = dataset[0]

        # contact_faces_original_nm should contain pre-normalization coordinates
        assert "contact_faces_original_nm" in sample
        np.testing.assert_array_almost_equal(
            sample["contact_faces_original_nm"][0, 0].numpy(), original_faces[0]
        )


def test_seg_contact_dataset_affinity_noise():
    """Test that affinity noise is applied to contact-face points."""
    with tempfile.TemporaryDirectory() as temp_dir:
        layer = make_layer_with_contacts(temp_dir)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(
            layer=layer,
            sample_indexer=indexer,
            merge_decision_authority="human",
            include_contact_faces_in_pointcloud=True,
            contact_label=0.0,
            affinity_channel_mode="per_point",
            affinity_noise_std=0.05,
            affinity_noise_prob=1.0,  # always apply
        )

        sample = dataset[0]

        assert "pointcloud" in sample
        assert sample["pointcloud"].shape[2] == 5


def test_seg_contact_dataset_channel_masking():
    """Test that channel masking is applied through the dataset pipeline."""
    with tempfile.TemporaryDirectory() as temp_dir:
        layer = make_layer_with_contacts(temp_dir)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(
            layer=layer,
            sample_indexer=indexer,
            merge_decision_authority="human",
            include_contact_faces_in_pointcloud=True,
            contact_label=0.0,
            affinity_channel_mode="per_point",
            mask_channel_probs=[1.0, 1.0],  # always mask both feature channels
            mask_mode=["global", "random"],
            mask_mode_global_prob=[1.0, 0.2],
            mask_value=0.0,
        )

        sample = dataset[0]

        assert "pointcloud" in sample
        pc = sample["pointcloud"]
        assert pc.shape[2] == 5
        # Channel 3 (segment label) should be masked to 0 (global mode, prob=1.0)
        assert (pc[:, :, 3] == 0.0).all()


def test_seg_contact_dataset_missing_config_key_skips():
    """Test contact is skipped when it has local_pointclouds but missing the required config."""
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
            "local_point_clouds": [{"radius_nm": 500, "n_points": 64}, {"radius_nm": 2000, "n_points": 128}],
            "merge_decisions": ["human"],
        }
        with open(f"{temp_dir}/info", "w", encoding="utf-8") as f:
            json.dump(info, f)

        contacts = [
            # Has both configs — sets config_keys to [(500,64), (2000,128)]
            SegContact(
                id=1,
                seg_a=100,
                seg_b=200,
                com=Vec3D(100.0, 100.0, 100.0),
                contact_faces=np.array([[1, 2, 3, 0.5]], dtype=np.float32),
                representative_points={100: Vec3D(90.0, 90.0, 90.0), 200: Vec3D(110.0, 110.0, 110.0)},
                local_pointclouds={
                    (500, 64): {
                        100: np.random.randn(64, 3).astype(np.float32),
                        200: np.random.randn(64, 3).astype(np.float32),
                    },
                    (2000, 128): {
                        100: np.random.randn(128, 3).astype(np.float32),
                        200: np.random.randn(128, 3).astype(np.float32),
                    },
                },
                merge_decisions={"human": True},
            ),
            # Has only (500,64), missing (2000,128) — should be skipped
            SegContact(
                id=2,
                seg_a=300,
                seg_b=400,
                com=Vec3D(120.0, 120.0, 100.0),
                contact_faces=np.array([[4, 5, 6, 0.8]], dtype=np.float32),
                representative_points={300: Vec3D(110.0, 110.0, 90.0), 400: Vec3D(130.0, 130.0, 110.0)},
                local_pointclouds={(500, 64): {
                    300: np.random.randn(64, 3).astype(np.float32),
                    400: np.random.randn(64, 3).astype(np.float32),
                }},
                merge_decisions={"human": False},
            ),
        ]
        backend.write_chunk((0, 0, 0), contacts)

        layer = VolumetricSegContactLayer(backend=backend)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(
            layer=layer, sample_indexer=indexer, merge_decision_authority="human",
        )

        sample = dataset[0]

        # Only contact 1 should pass (has both configs); contact 2 is missing (2000,128)
        assert sample["contact_id"].shape[0] == 1
        assert sample["contact_id"][0].item() == 1


def test_seg_contact_dataset_all_contacts_missing_pointclouds_returns_empty():
    """When all contacts pass affinity filter but have no pointclouds, return {}."""
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
                contact_faces=np.array([[1, 2, 3, 0.3]], dtype=np.float32),
                representative_points={100: Vec3D(90.0, 90.0, 90.0), 200: Vec3D(110.0, 110.0, 110.0)},
                local_pointclouds=None,
                merge_decisions={"human": True},
            ),
        ]
        backend.write_chunk((0, 0, 0), contacts)

        layer = VolumetricSegContactLayer(backend=backend)
        indexer = SegContactIndexer(path=temp_dir)
        dataset = SegContactDataset(
            layer=layer, sample_indexer=indexer, merge_decision_authority="human",
        )

        sample = dataset[0]
        assert sample == {}

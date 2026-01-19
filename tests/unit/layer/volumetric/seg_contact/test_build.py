import json
import os
import tempfile

import pytest

from zetta_utils import builder
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric.seg_contact import (
    SegContactInfoSpec,
    SegContactInfoSpecParams,
    SegContactLayerBackend,
    VolumetricSegContactLayer,
    build_seg_contact_info_spec,
)
from zetta_utils.layer.volumetric.seg_contact.build import build_seg_contact_layer


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


# --- Read mode tests ---


def test_build_seg_contact_layer_read_mode():
    """Test building a contact layer in read mode."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_backend(temp_dir)

        layer = build_seg_contact_layer(path=temp_dir, mode="read")

        assert isinstance(layer, VolumetricSegContactLayer)
        assert layer.readonly is True


def test_build_seg_contact_layer_read_nonexistent():
    """Test that read mode fails for nonexistent path."""
    with pytest.raises(FileNotFoundError):
        build_seg_contact_layer(path="/path/to/nonexistent", mode="read")


# --- Update mode tests ---


def test_build_seg_contact_layer_update_mode():
    """Test building a contact layer in update mode."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_backend(temp_dir)

        layer = build_seg_contact_layer(path=temp_dir, mode="update")

        assert isinstance(layer, VolumetricSegContactLayer)
        assert layer.readonly is False


def test_build_seg_contact_layer_update_nonexistent():
    """Test that update mode fails for nonexistent path."""
    with pytest.raises(FileNotFoundError):
        build_seg_contact_layer(path="/path/to/nonexistent", mode="update")


def test_build_seg_contact_layer_preserves_backend_properties():
    """Test that built layer has correct backend properties."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = make_backend(temp_dir)

        layer = build_seg_contact_layer(path=temp_dir, mode="read")

        assert layer.backend.resolution == backend.resolution
        assert layer.backend.voxel_offset == backend.voxel_offset
        assert layer.backend.size == backend.size
        assert layer.backend.chunk_size == backend.chunk_size
        assert layer.backend.max_contact_span == backend.max_contact_span


# --- Write mode tests ---


def test_build_seg_contact_layer_write_mode():
    """Test creating a new contact layer in write mode."""
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "new_layer")
        bbox = BBox3D.from_coords([0, 0, 0], [16000, 16000, 20000], [16, 16, 40])
        info_spec = SegContactInfoSpec(
            info_spec_params=SegContactInfoSpecParams(
                resolution=Vec3D(16, 16, 40),
                chunk_size=Vec3D(256, 256, 128),
                max_contact_span=512,
                bbox=bbox,
            )
        )

        layer = build_seg_contact_layer(path=path, mode="write", info_spec=info_spec)

        assert isinstance(layer, VolumetricSegContactLayer)
        assert layer.readonly is False
        assert os.path.exists(os.path.join(path, "info"))


def test_build_seg_contact_layer_write_no_info_spec_fails():
    """Test that write mode fails without info_spec."""
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "new_layer")

        with pytest.raises(ValueError, match="info_spec is required"):
            build_seg_contact_layer(path=path, mode="write")


def test_build_seg_contact_layer_write_existing_fails():
    """Test that write mode fails when layer exists without info_overwrite."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_backend(temp_dir)
        bbox = BBox3D.from_coords([0, 0, 0], [16000, 16000, 20000], [16, 16, 40])
        info_spec = SegContactInfoSpec(
            info_spec_params=SegContactInfoSpecParams(
                resolution=Vec3D(16, 16, 40),
                chunk_size=Vec3D(256, 256, 128),
                max_contact_span=512,
                bbox=bbox,
            )
        )

        with pytest.raises(RuntimeError):
            build_seg_contact_layer(path=temp_dir, mode="write", info_spec=info_spec)


def test_build_seg_contact_layer_write_overwrite():
    """Test that write mode succeeds with info_overwrite=True."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_backend(temp_dir)
        bbox = BBox3D.from_coords([0, 0, 0], [32000, 32000, 40000], [16, 16, 40])
        info_spec = SegContactInfoSpec(
            info_spec_params=SegContactInfoSpecParams(
                resolution=Vec3D(16, 16, 40),
                chunk_size=Vec3D(512, 512, 256),
                max_contact_span=1024,
                bbox=bbox,
            )
        )

        layer = build_seg_contact_layer(
            path=temp_dir, mode="write", info_spec=info_spec, info_overwrite=True
        )

        assert isinstance(layer, VolumetricSegContactLayer)
        assert layer.backend.chunk_size == Vec3D(512, 512, 256)
        assert layer.backend.max_contact_span == 1024


# --- Builder system tests ---


def test_build_seg_contact_layer_via_builder_read():
    """Test building contact layer via builder system in read mode."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_backend(temp_dir)

        spec = {
            "@type": "build_seg_contact_layer",
            "path": temp_dir,
            "mode": "read",
        }

        layer = builder.build(spec)

        assert isinstance(layer, VolumetricSegContactLayer)
        assert layer.readonly is True


def test_build_seg_contact_layer_via_builder_update():
    """Test building contact layer via builder system in update mode."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_backend(temp_dir)

        spec = {
            "@type": "build_seg_contact_layer",
            "path": temp_dir,
            "mode": "update",
        }

        layer = builder.build(spec)

        assert isinstance(layer, VolumetricSegContactLayer)
        assert layer.readonly is False


def test_build_seg_contact_layer_via_builder_write():
    """Test building new contact layer via builder system."""
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "new_layer")

        spec = {
            "@type": "build_seg_contact_layer",
            "path": path,
            "mode": "write",
            "info_spec": {
                "@type": "build_seg_contact_info_spec",
                "resolution": [16, 16, 40],
                "chunk_size": [256, 256, 128],
                "max_contact_span": 512,
                "bbox": {
                    "@type": "BBox3D.from_coords",
                    "start_coord": [0, 0, 0],
                    "end_coord": [1000, 1000, 500],
                    "resolution": [16, 16, 40],
                },
            },
        }

        layer = builder.build(spec)

        assert isinstance(layer, VolumetricSegContactLayer)
        assert layer.readonly is False


# --- SegContactInfoSpec tests ---


def test_contact_info_spec_from_params():
    """Test creating SegContactInfoSpec from params."""
    bbox = BBox3D.from_coords([0, 0, 0], [16000, 16000, 20000], [16, 16, 40])
    spec = SegContactInfoSpec(
        info_spec_params=SegContactInfoSpecParams(
            resolution=Vec3D(16, 16, 40),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
            bbox=bbox,
        )
    )

    info = spec.make_info()

    assert info["type"] == "seg_contact"
    assert info["resolution"] == [16, 16, 40]
    assert info["chunk_size"] == [256, 256, 128]
    assert info["max_contact_span"] == 512


def test_contact_info_spec_from_path():
    """Test creating SegContactInfoSpec from existing path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_backend(temp_dir)

        spec = SegContactInfoSpec(info_path=temp_dir)
        info = spec.make_info()

        assert info["type"] == "seg_contact"
        assert info["resolution"] == [16, 16, 40]


def test_contact_info_spec_write_info():
    """Test writing info file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "new_layer")
        bbox = BBox3D.from_coords([0, 0, 0], [16000, 16000, 20000], [16, 16, 40])
        spec = SegContactInfoSpec(
            info_spec_params=SegContactInfoSpecParams(
                resolution=Vec3D(16, 16, 40),
                chunk_size=Vec3D(256, 256, 128),
                max_contact_span=512,
                bbox=bbox,
            )
        )

        spec.write_info(path)

        assert os.path.exists(os.path.join(path, "info"))
        # Verify we can load it back
        backend = SegContactLayerBackend.from_path(path)
        assert backend.resolution == Vec3D(16, 16, 40)


def test_build_seg_contact_info_spec_from_params():
    """Test builder function with direct params."""
    bbox = BBox3D.from_coords([0, 0, 0], [16000, 16000, 20000], [16, 16, 40])
    spec = build_seg_contact_info_spec(
        resolution=[16, 16, 40],
        chunk_size=[256, 256, 128],
        max_contact_span=512,
        bbox=bbox,
    )

    assert spec.info_spec_params is not None
    assert spec.info_spec_params.resolution == Vec3D(16, 16, 40)


def test_build_seg_contact_info_spec_from_reference():
    """Test builder function with source path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_backend(temp_dir)

        spec = build_seg_contact_info_spec(source_path=temp_dir)

        assert spec.info_spec_params is not None
        assert spec.info_spec_params.resolution == Vec3D(16, 16, 40)


# --- Additional coverage tests ---


def test_build_seg_contact_layer_invalid_mode():
    """Test that invalid mode raises ValueError."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_backend(temp_dir)

        with pytest.raises(ValueError, match="Invalid mode"):
            build_seg_contact_layer(path=temp_dir, mode="invalid")  # type: ignore


def test_seg_contact_info_spec_both_none_raises():
    """Test that providing neither info_path nor info_spec_params raises."""
    with pytest.raises(ValueError, match="Exactly one"):
        SegContactInfoSpec(info_path=None, info_spec_params=None)


def test_seg_contact_info_spec_both_provided_raises():
    """Test that providing both info_path and info_spec_params raises."""
    bbox = BBox3D.from_coords([0, 0, 0], [16000, 16000, 20000], [16, 16, 40])
    params = SegContactInfoSpecParams(
        resolution=Vec3D(16, 16, 40),
        chunk_size=Vec3D(256, 256, 128),
        max_contact_span=512,
        bbox=bbox,
    )
    with pytest.raises(ValueError, match="Exactly one"):
        SegContactInfoSpec(info_path="/some/path", info_spec_params=params)


def test_seg_contact_info_spec_set_bbox():
    """Test set_bbox method updates the bounding box."""
    bbox = BBox3D.from_coords([0, 0, 0], [16000, 16000, 20000], [16, 16, 40])
    spec = SegContactInfoSpec(
        info_spec_params=SegContactInfoSpecParams(
            resolution=Vec3D(16, 16, 40),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
            bbox=bbox,
        )
    )

    new_bbox = BBox3D.from_coords([0, 0, 0], [32000, 32000, 40000], [16, 16, 40])
    spec.set_bbox(new_bbox)

    assert spec.info_spec_params is not None
    assert spec.info_spec_params.bbox == new_bbox


def test_seg_contact_info_spec_make_info_with_optional_paths():
    """Test make_info includes optional paths when specified."""
    bbox = BBox3D.from_coords([0, 0, 0], [16000, 16000, 20000], [16, 16, 40])
    spec = SegContactInfoSpec(
        info_spec_params=SegContactInfoSpecParams(
            resolution=Vec3D(16, 16, 40),
            chunk_size=Vec3D(256, 256, 128),
            max_contact_span=512,
            bbox=bbox,
            segmentation_path="gs://bucket/seg",
            ground_truth_path="gs://bucket/gt",
            affinity_path="gs://bucket/aff",
            image_path="gs://bucket/img",
            local_point_clouds=[{"radius_nm": 500, "n_points": 64}],
            merge_decisions=["human"],
            merge_probabilities=["model_v1"],
            filter_settings={"min_affinity": 0.5},
        )
    )

    info = spec.make_info()

    assert info["segmentation_path"] == "gs://bucket/seg"
    assert info["ground_truth_path"] == "gs://bucket/gt"
    assert info["affinity_path"] == "gs://bucket/aff"
    assert info["image_path"] == "gs://bucket/img"
    assert info["local_point_clouds"] == [{"radius_nm": 500, "n_points": 64}]
    assert info["merge_decisions"] == ["human"]
    assert info["merge_probabilities"] == ["model_v1"]
    assert info["filter_settings"] == {"min_affinity": 0.5}


def test_build_seg_contact_info_spec_info_path_with_other_params_raises():
    """Test that info_path with other params raises ValueError."""
    with pytest.raises(ValueError, match="When `info_path` is provided"):
        build_seg_contact_info_spec(
            info_path="/some/path",
            resolution=[16, 16, 40],
        )


def test_build_seg_contact_info_spec_missing_required_params_raises():
    """Test that missing required params without source raises ValueError."""
    with pytest.raises(ValueError, match="resolution, chunk_size"):
        build_seg_contact_info_spec(
            resolution=[16, 16, 40],
            # Missing chunk_size, max_contact_span, bbox
        )


def test_build_seg_contact_info_spec_from_info_path_only():
    """Test builder function with only info_path (no other params)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_backend(temp_dir)

        spec = build_seg_contact_info_spec(info_path=temp_dir)

        assert spec.info_path == temp_dir
        assert spec.info_spec_params is None
        # Verify make_info works
        info = spec.make_info()
        assert info["type"] == "seg_contact"


def test_build_seg_contact_layer_read_with_local_point_clouds():
    """Test read mode with local_point_clouds filter."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_backend(temp_dir)

        layer = build_seg_contact_layer(path=temp_dir, mode="read", local_point_clouds=[(500, 64)])

        assert layer.backend.local_point_clouds == [(500, 64)]


def test_build_seg_contact_layer_write_with_local_point_clouds():
    """Test write mode with local_point_clouds filter."""
    with tempfile.TemporaryDirectory() as temp_dir:
        bbox = BBox3D.from_coords([0, 0, 0], [32000, 32000, 40000], [16, 16, 40])
        info_spec = SegContactInfoSpec(
            info_spec_params=SegContactInfoSpecParams(
                resolution=Vec3D(16, 16, 40),
                chunk_size=Vec3D(256, 256, 128),
                max_contact_span=512,
                bbox=bbox,
            )
        )

        layer = build_seg_contact_layer(
            path=temp_dir,
            mode="write",
            info_spec=info_spec,
            local_point_clouds=[(500, 64)],
        )

        assert layer.backend.local_point_clouds == [(500, 64)]


def test_build_seg_contact_layer_update_with_info_spec():
    """Test update mode with info_spec to update info file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create initial layer
        make_backend(temp_dir)

        # Update with new info_spec
        bbox = BBox3D.from_coords([0, 0, 0], [64000, 64000, 80000], [16, 16, 40])
        info_spec = SegContactInfoSpec(
            info_spec_params=SegContactInfoSpecParams(
                resolution=Vec3D(16, 16, 40),
                chunk_size=Vec3D(256, 256, 128),
                max_contact_span=512,
                bbox=bbox,
                local_point_clouds=[{"radius_nm": 1000, "n_points": 128}],
            )
        )

        layer = build_seg_contact_layer(
            path=temp_dir,
            mode="update",
            info_spec=info_spec,
            info_overwrite=True,
        )

        assert layer is not None
        # Verify info was updated
        with open(f"{temp_dir}/info", encoding="utf-8") as f:
            info = json.load(f)
        assert info["local_point_clouds"] == [{"radius_nm": 1000, "n_points": 128}]


def test_build_seg_contact_layer_update_with_local_point_clouds():
    """Test update mode with local_point_clouds filter."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_backend(temp_dir)

        layer = build_seg_contact_layer(
            path=temp_dir, mode="update", local_point_clouds=[(500, 64)]
        )

        assert layer.backend.local_point_clouds == [(500, 64)]


def test_update_info_removing_pointcloud_config_fails_without_overwrite():
    """Test that removing pointcloud configs fails without info_overwrite=True."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create initial layer with pointcloud config
        bbox = BBox3D.from_coords([0, 0, 0], [64000, 64000, 80000], [16, 16, 40])
        info_spec = SegContactInfoSpec(
            info_spec_params=SegContactInfoSpecParams(
                resolution=Vec3D(16, 16, 40),
                chunk_size=Vec3D(256, 256, 128),
                max_contact_span=512,
                bbox=bbox,
                local_point_clouds=[
                    {"radius_nm": 500, "n_points": 64},
                    {"radius_nm": 1000, "n_points": 128},
                ],
            )
        )
        build_seg_contact_layer(path=temp_dir, mode="write", info_spec=info_spec)

        # Try to update with fewer pointcloud configs (removing one) without overwrite
        new_info_spec = SegContactInfoSpec(
            info_spec_params=SegContactInfoSpecParams(
                resolution=Vec3D(16, 16, 40),
                chunk_size=Vec3D(256, 256, 128),
                max_contact_span=512,
                bbox=bbox,
                local_point_clouds=[{"radius_nm": 500, "n_points": 64}],  # Missing (1000, 128)
            )
        )

        with pytest.raises(RuntimeError, match="Some pointcloud configs"):
            build_seg_contact_layer(
                path=temp_dir,
                mode="update",
                info_spec=new_info_spec,
                info_overwrite=False,  # Should fail because we're removing a config
                info_keep_existing_pointcloud_configs=False,  # Don't merge, so removal is detected
            )

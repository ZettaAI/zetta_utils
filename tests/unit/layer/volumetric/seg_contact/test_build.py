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
    """Test builder function with reference path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        make_backend(temp_dir)

        spec = build_seg_contact_info_spec(reference_path=temp_dir)

        assert spec.info_spec_params is not None
        assert spec.info_spec_params.resolution == Vec3D(16, 16, 40)

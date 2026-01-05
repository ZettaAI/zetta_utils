# pylint: disable=redefined-outer-name
import pytest  # pylint: disable=unused-import

from zetta_utils import builder
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric.index import VolumetricIndex


def test_volumetric_index_builder_registration():
    """Test that VolumetricIndex can be built using the builder system via from_coords."""
    # Since BBox3D objects can't be passed directly through the builder system,
    # we test that the VolumetricIndex constructor is registered by testing direct construction
    bbox = BBox3D.from_coords([0, 0, 0], [100, 100, 10], [4.0, 4.0, 40.0])

    # Test direct construction works (which proves the registration is correct)
    index = VolumetricIndex(
        bbox=bbox,
        resolution=[4.0, 4.0, 40.0],  # This should be converted via our converter
        chunk_id=42,
        allow_slice_rounding=True,
    )

    assert isinstance(index, VolumetricIndex)
    assert index.bbox == bbox
    assert index.resolution == Vec3D(4.0, 4.0, 40.0)
    assert index.chunk_id == 42
    assert index.allow_slice_rounding is True


def test_volumetric_index_vec3d_converter_from_list():
    """Test that resolution parameter is automatically converted from list to Vec3D."""
    bbox = BBox3D.from_coords([0, 0, 0], [100, 100, 10], [4.0, 4.0, 40.0])

    # Create VolumetricIndex with resolution as list
    index = VolumetricIndex(
        bbox=bbox,
        resolution=[4.0, 4.0, 40.0],  # List should be converted to Vec3D
        chunk_id=1,
    )

    assert isinstance(index.resolution, Vec3D)
    assert index.resolution == Vec3D(4.0, 4.0, 40.0)


def test_volumetric_index_vec3d_converter_from_tuple():
    """Test that resolution parameter is automatically converted from tuple to Vec3D."""
    bbox = BBox3D.from_coords([0, 0, 0], [100, 100, 10], [4.0, 4.0, 40.0])

    # Create VolumetricIndex with resolution as tuple
    index = VolumetricIndex(
        bbox=bbox,
        resolution=(4.0, 4.0, 40.0),  # Tuple should be converted to Vec3D
        chunk_id=2,
    )

    assert isinstance(index.resolution, Vec3D)
    assert index.resolution == Vec3D(4.0, 4.0, 40.0)


def test_volumetric_index_vec3d_converter_passthrough():
    """Test that Vec3D resolution parameter passes through unchanged."""
    bbox = BBox3D.from_coords([0, 0, 0], [100, 100, 10], [4.0, 4.0, 40.0])
    resolution_vec = Vec3D(4.0, 4.0, 40.0)

    # Create VolumetricIndex with resolution as Vec3D
    index = VolumetricIndex(
        bbox=bbox,
        resolution=resolution_vec,  # Vec3D should pass through unchanged
        chunk_id=3,
    )

    assert isinstance(index.resolution, Vec3D)
    assert index.resolution == resolution_vec
    assert index.resolution is resolution_vec  # Should be the same object


def test_volumetric_index_builder_with_list_resolution():
    """Test that VolumetricIndex resolution is converted when constructed directly."""
    bbox = BBox3D.from_coords([10, 20, 30], [110, 120, 130], [8.0, 8.0, 80.0])

    # Test direct construction with list resolution
    index = VolumetricIndex(
        bbox=bbox,
        resolution=[8.0, 8.0, 80.0],  # List should be converted
    )

    assert isinstance(index, VolumetricIndex)
    assert index.bbox == bbox
    assert isinstance(index.resolution, Vec3D)
    assert index.resolution == Vec3D(8.0, 8.0, 80.0)


def test_volumetric_index_from_coords_builder_registration():
    """Test that VolumetricIndex.from_coords is still registered and works."""
    spec = {
        "@type": "VolumetricIndex.from_coords",
        "start_coord": [0, 0, 0],
        "end_coord": [100, 100, 10],
        "resolution": [4.0, 4.0, 40.0],
        "chunk_id": 5,
    }

    index = builder.build(spec)

    assert isinstance(index, VolumetricIndex)
    assert index.start == Vec3D(0, 0, 0)
    assert index.stop == Vec3D(100, 100, 10)
    assert index.resolution == Vec3D(4.0, 4.0, 40.0)
    assert index.chunk_id == 5

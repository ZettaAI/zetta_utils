# pylint: disable=redefined-outer-name
import os
import shutil
import tempfile

import pytest

from zetta_utils import builder
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric.annotation.annotations import (
    PropertySpec,
    Relationship,
)
from zetta_utils.layer.volumetric.annotation.build import build_annotation_layer
from zetta_utils.layer.volumetric.annotation.layer import VolumetricAnnotationLayer
from zetta_utils.layer.volumetric.index import VolumetricIndex


@pytest.fixture
def temp_dir():
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


def test_build_annotation_layer_write_mode(temp_dir):
    path = os.path.join(temp_dir, "annotations")

    layer = build_annotation_layer(
        path=path,
        annotation_type="LINE",
        resolution=[4.0, 4.0, 40.0],
        dataset_size=[1000, 1000, 100],
        voxel_offset=[0, 0, 0],
        mode="write",
        default_desired_resolution=[4.0, 4.0, 40.0],
        index_resolution=[4.0, 4.0, 40.0],
        allow_slice_rounding=True,
    )

    assert isinstance(layer, VolumetricAnnotationLayer)
    assert layer.readonly is False
    assert layer.allow_slice_rounding is True
    assert layer.default_desired_resolution == Vec3D(4.0, 4.0, 40.0)
    assert layer.index_resolution == Vec3D(4.0, 4.0, 40.0)


def test_build_annotation_layer_with_index(temp_dir):
    path = os.path.join(temp_dir, "annotations_with_index")

    index = VolumetricIndex.from_coords([0, 0, 0], [1000, 1000, 100], [4.0, 4.0, 40.0])

    layer = build_annotation_layer(
        path=path,
        annotation_type="LINE",
        index=index,
        mode="write",
    )

    assert isinstance(layer, VolumetricAnnotationLayer)
    assert layer.readonly is False


def test_build_annotation_layer_with_processors(temp_dir):
    path = os.path.join(temp_dir, "annotations_with_processors")

    index = VolumetricIndex.from_coords([0, 0, 0], [1000, 1000, 100], [4.0, 4.0, 40.0])

    def index_proc(idx):
        return idx

    def read_proc(data):
        return data

    def write_proc(data):
        return data

    layer = build_annotation_layer(
        path=path,
        annotation_type="LINE",
        index=index,
        mode="write",
        index_procs=[index_proc],
        read_procs=[read_proc],
        write_procs=[write_proc],
    )

    assert isinstance(layer, VolumetricAnnotationLayer)
    assert len(layer.index_procs) == 1
    assert len(layer.read_procs) == 1
    assert len(layer.write_procs) == 1


def test_build_annotation_layer_with_custom_chunk_sizes(temp_dir):
    path = os.path.join(temp_dir, "annotations_with_chunks")

    index = VolumetricIndex.from_coords([0, 0, 0], [1000, 1000, 100], [4.0, 4.0, 40.0])
    chunk_sizes = [[500, 500, 50], [250, 250, 25]]

    layer = build_annotation_layer(
        path=path,
        annotation_type="LINE",
        index=index,
        mode="write",
        chunk_sizes=chunk_sizes,
    )

    assert isinstance(layer, VolumetricAnnotationLayer)


def test_build_annotation_layer_replace_mode(temp_dir):
    path = os.path.join(temp_dir, "annotations_to_replace")

    # First create a layer
    build_annotation_layer(
        path=path,
        annotation_type="LINE",
        resolution=[4.0, 4.0, 40.0],
        dataset_size=[1000, 1000, 100],
        voxel_offset=[0, 0, 0],
        mode="write",
    )

    # Then replace it
    layer = build_annotation_layer(
        path=path,
        mode="replace",
        resolution=[4.0, 4.0, 40.0],
        dataset_size=[1000, 1000, 100],
        voxel_offset=[0, 0, 0],
    )

    assert isinstance(layer, VolumetricAnnotationLayer)
    assert layer.readonly is False


def test_build_annotation_layer_update_mode(temp_dir):
    path = os.path.join(temp_dir, "annotations_to_update")

    layer = build_annotation_layer(
        path=path,
        annotation_type="LINE",
        resolution=[4.0, 4.0, 40.0],
        dataset_size=[1000, 1000, 100],
        voxel_offset=[0, 0, 0],
        index_resolution=[4.0, 4.0, 40.0],
        default_desired_resolution=[4.0, 4.0, 40.0],
        mode="write",
    )
    layer[0:10, 0:10, 0:10] = []

    layer = build_annotation_layer(
        path=path,
        mode="update",
    )

    assert isinstance(layer, VolumetricAnnotationLayer)
    assert layer.readonly is False


def test_build_annotation_layer_read_mode(temp_dir):
    path = os.path.join(temp_dir, "annotations_to_read")

    layer = build_annotation_layer(
        path=path,
        annotation_type="LINE",
        resolution=[4.0, 4.0, 40.0],
        dataset_size=[1000, 1000, 100],
        index_resolution=[4.0, 4.0, 40.0],
        default_desired_resolution=[4.0, 4.0, 40.0],
        voxel_offset=[0, 0, 0],
        mode="write",
    )
    layer[0:10, 0:10, 0:10] = []

    layer = build_annotation_layer(
        path=path,
        mode="read",
    )
    assert isinstance(layer, VolumetricAnnotationLayer)
    assert layer.readonly is True


def test_build_annotation_layer_read_mode_nonexistent_file():
    with pytest.raises(Exception):
        build_annotation_layer(
            path="/path/to/nonexistent",
            mode="read",
        )


def test_build_annotation_layer_update_mode_nonexistent_file():
    with pytest.raises(Exception):
        build_annotation_layer(
            path="/path/to/nonexistent",
            mode="update",
        )


def test_build_annotation_layer_write_mode_existing_file(temp_dir):
    path = os.path.join(temp_dir, "existing_annotations")

    # First create a layer
    build_annotation_layer(
        path=path,
        annotation_type="LINE",
        resolution=[4.0, 4.0, 40.0],
        dataset_size=[1000, 1000, 100],
        voxel_offset=[0, 0, 0],
        mode="write",
    )

    # Then try to write to it again
    with pytest.raises(Exception):
        build_annotation_layer(
            path=path,
            resolution=[4.0, 4.0, 40.0],
            dataset_size=[1000, 1000, 100],
            voxel_offset=[0, 0, 0],
            mode="write",
        )


def test_build_annotation_layer_missing_resolution():
    with pytest.raises(Exception):
        build_annotation_layer(
            path="/path/to/annotations",
            dataset_size=[1000, 1000, 100],
            voxel_offset=[0, 0, 0],
            mode="write",
        )


def test_build_annotation_layer_missing_dataset_size():
    with pytest.raises(Exception):
        build_annotation_layer(
            path="/path/to/annotations",
            annotation_type="LINE",
            resolution=[4.0, 4.0, 40.0],
            voxel_offset=[0, 0, 0],
            mode="write",
        )


def test_build_annotation_layer_missing_voxel_offset():
    with pytest.raises(Exception):
        build_annotation_layer(
            path="/path/to/annotations",
            resolution=[4.0, 4.0, 40.0],
            dataset_size=[1000, 1000, 100],
            mode="write",
        )


def test_build_annotation_layer_invalid_resolution_length():
    with pytest.raises(Exception):
        build_annotation_layer(
            path="/path/to/annotations",
            annotation_type="LINE",
            resolution=[4.0, 4.0],
            dataset_size=[1000, 1000, 100],
            voxel_offset=[0, 0, 0],
            mode="write",
        )


def test_build_annotation_layer_invalid_dataset_size_length():
    with pytest.raises(Exception):
        build_annotation_layer(
            path="/path/to/annotations",
            resolution=[4.0, 4.0, 40.0],
            dataset_size=[1000, 1000],
            voxel_offset=[0, 0, 0],
            mode="write",
        )


def test_build_annotation_layer_invalid_voxel_offset_length():
    with pytest.raises(Exception):
        build_annotation_layer(
            path="/path/to/annotations",
            resolution=[4.0, 4.0, 40.0],
            dataset_size=[1000, 1000, 100],
            voxel_offset=[0, 0],
            mode="write",
        )


def test_build_annotation_layer_with_property_specs_and_relationships(temp_dir):
    """Test building annotation layer with property specs and relationships
    using builder system."""
    path = os.path.join(temp_dir, "annotations_with_props_and_rels")

    # Test CUE-like specification
    spec = {
        "@type": "build_annotation_layer",
        "path": path,
        "mode": "write",
        "annotation_type": "LINE",
        "resolution": [4, 4, 40],
        "dataset_size": [1000, 1000, 100],
        "voxel_offset": [0, 0, 0],
        "property_specs": [
            {
                "@type": "build_property_spec",
                "id": "score",
                "type": "float32",
                "description": "Confidence score",
            },
            {
                "@type": "build_property_spec",
                "id": "category",
                "type": "uint8",
                "description": "Object category",
                "enum_values": [0, 1, 2],
                "enum_labels": ["unknown", "synapse", "mitochondria"],
            },
        ],
        "relationships": [
            {"@type": "build_relationship", "id": "presyn_cell"},
            {"@type": "build_relationship", "id": "postsyn_cell"},
        ],
    }

    layer = builder.build(spec)

    # Verify layer was created correctly
    assert isinstance(layer, VolumetricAnnotationLayer)
    assert layer.readonly is False

    # Check that the backend has the property specs and relationships
    backend = layer.backend
    assert len(backend.property_specs) == 2
    assert len(backend.relationships) == 2

    # Verify property specs
    score_prop = backend.property_specs[0]
    assert score_prop.id == "score"
    assert score_prop.type == "float32"
    assert score_prop.description == "Confidence score"
    assert not score_prop.has_enums()

    category_prop = backend.property_specs[1]
    assert category_prop.id == "category"
    assert category_prop.type == "uint8"
    assert category_prop.description == "Object category"
    assert category_prop.has_enums()
    assert category_prop.enum_values == [0, 1, 2]
    assert category_prop.enum_labels == ["unknown", "synapse", "mitochondria"]

    # Verify relationships
    presyn_rel = backend.relationships[0]
    assert presyn_rel.id == "presyn_cell"
    assert presyn_rel.key == "presyncell"  # auto-generated from id

    postsyn_rel = backend.relationships[1]
    assert postsyn_rel.id == "postsyn_cell"
    assert postsyn_rel.key == "postsyncell"  # auto-generated from id


def test_build_annotation_layer_with_direct_property_specs_and_relationships(temp_dir):
    """Test building annotation layer with direct property specs and relationships."""
    path = os.path.join(temp_dir, "annotations_direct_props_and_rels")

    # Create property specs and relationships directly
    property_specs = [
        PropertySpec(id="confidence", type="float32", description="Detection confidence"),
        PropertySpec(id="color", type="rgb", description="Display color"),
    ]

    relationships = [Relationship(id="parent_cell"), Relationship(id="child_cell", key="child")]

    layer = build_annotation_layer(
        path=path,
        annotation_type="LINE",
        resolution=[4.0, 4.0, 40.0],
        dataset_size=[1000, 1000, 100],
        voxel_offset=[0, 0, 0],
        mode="write",
        property_specs=property_specs,
        relationships=relationships,
    )

    # Verify layer was created correctly
    assert isinstance(layer, VolumetricAnnotationLayer)
    assert layer.readonly is False

    # Check that the backend has the property specs and relationships
    backend = layer.backend
    assert len(backend.property_specs) == 2
    assert len(backend.relationships) == 2

    # Verify property specs
    confidence_prop = backend.property_specs[0]
    assert confidence_prop.id == "confidence"
    assert confidence_prop.type == "float32"
    assert confidence_prop.description == "Detection confidence"

    color_prop = backend.property_specs[1]
    assert color_prop.id == "color"
    assert color_prop.type == "rgb"
    assert color_prop.description == "Display color"

    # Verify relationships
    parent_rel = backend.relationships[0]
    assert parent_rel.id == "parent_cell"
    assert parent_rel.key == "parentcell"  # auto-generated from id

    child_rel = backend.relationships[1]
    assert child_rel.id == "child_cell"
    assert child_rel.key == "child"  # explicitly provided


def test_build_annotation_layer_with_bbox_and_resolution(temp_dir):
    """Test building annotation layer with bbox and resolution.

    Uses bbox and resolution instead of dataset_size/voxel_offset.
    """
    path = os.path.join(temp_dir, "annotations_bbox_resolution")

    # Create a BBox3D
    bbox = BBox3D.from_coords([10, 20, 5], [110, 120, 15], [4.0, 4.0, 40.0])

    layer = build_annotation_layer(
        path=path,
        annotation_type="POINT",
        bbox=bbox,
        resolution=[4.0, 4.0, 40.0],
        mode="write",
    )

    assert isinstance(layer, VolumetricAnnotationLayer)
    assert layer.readonly is False

    # Verify the backend has the correct index
    backend = layer.backend
    assert backend.index is not None
    expected_index = VolumetricIndex(bbox=bbox, resolution=Vec3D(4.0, 4.0, 40.0))
    assert backend.index.bbox == expected_index.bbox
    assert backend.index.resolution == expected_index.resolution


def test_build_annotation_layer_bbox_resolution_via_builder(temp_dir):
    """Test building annotation layer with bbox and resolution via builder system."""
    path = os.path.join(temp_dir, "annotations_bbox_resolution_builder")

    # Create a BBox3D and use it directly with build_annotation_layer function
    # (BBox3D objects can't be passed through the general builder system)
    bbox = BBox3D.from_coords([0, 0, 0], [200, 200, 20], [8.0, 8.0, 80.0])

    # Test the build_annotation_layer function directly with builder patterns
    layer = build_annotation_layer(
        path=path,
        annotation_type="LINE",
        bbox=bbox,
        resolution=[8.0, 8.0, 80.0],
        mode="write",
    )

    assert isinstance(layer, VolumetricAnnotationLayer)
    assert layer.readonly is False

    # Verify the backend has the correct index
    backend = layer.backend
    assert backend.index is not None
    expected_index = VolumetricIndex(bbox=bbox, resolution=Vec3D(8.0, 8.0, 80.0))
    assert backend.index.bbox == expected_index.bbox
    assert backend.index.resolution == expected_index.resolution


def test_build_annotation_layer_bbox_without_resolution():
    """Test that providing bbox without resolution raises an error."""
    bbox = BBox3D.from_coords([0, 0, 0], [100, 100, 10], [4.0, 4.0, 40.0])

    with pytest.raises(ValueError, match="when `bbox` is provided, `resolution` is also required"):
        build_annotation_layer(
            path="/path/to/annotations",
            bbox=bbox,
            mode="write",
        )


def test_build_annotation_layer_bbox_with_dataset_size_voxel_offset():
    """Test that providing bbox with dataset_size or voxel_offset raises an error."""
    bbox = BBox3D.from_coords([0, 0, 0], [100, 100, 10], [4.0, 4.0, 40.0])

    # Test with dataset_size
    with pytest.raises(
        ValueError,
        match=(
            "when `bbox` and `resolution` are provided, `dataset_size` and "
            "`voxel_offset` should not be provided"
        ),
    ):
        build_annotation_layer(
            path="/path/to/annotations",
            bbox=bbox,
            resolution=[4.0, 4.0, 40.0],
            dataset_size=[100, 100, 10],
            mode="write",
        )

    # Test with voxel_offset
    with pytest.raises(
        ValueError,
        match=(
            "when `bbox` and `resolution` are provided, `dataset_size` and "
            "`voxel_offset` should not be provided"
        ),
    ):
        build_annotation_layer(
            path="/path/to/annotations",
            bbox=bbox,
            resolution=[4.0, 4.0, 40.0],
            voxel_offset=[0, 0, 0],
            mode="write",
        )

    # Test with both
    with pytest.raises(
        ValueError,
        match=(
            "when `bbox` and `resolution` are provided, `dataset_size` and "
            "`voxel_offset` should not be provided"
        ),
    ):
        build_annotation_layer(
            path="/path/to/annotations",
            bbox=bbox,
            resolution=[4.0, 4.0, 40.0],
            dataset_size=[100, 100, 10],
            voxel_offset=[0, 0, 0],
            mode="write",
        )


def test_build_annotation_layer_bbox_resolution_invalid_resolution_length():
    """Test that providing bbox with invalid resolution length raises an error."""
    bbox = BBox3D.from_coords([0, 0, 0], [100, 100, 10], [4.0, 4.0, 40.0])

    with pytest.raises(ValueError, match="`resolution` needs 3 elements, not 2"):
        build_annotation_layer(
            path="/path/to/annotations",
            bbox=bbox,
            resolution=[4.0, 4.0],  # Only 2 elements
            mode="write",
        )


def test_build_annotation_layer_bbox_resolution_replace_mode(temp_dir):
    """Test bbox + resolution works with replace mode."""
    path = os.path.join(temp_dir, "annotations_bbox_replace")

    # First create a layer with traditional parameters
    build_annotation_layer(
        path=path,
        annotation_type="POINT",
        resolution=[4.0, 4.0, 40.0],
        dataset_size=[100, 100, 10],
        voxel_offset=[0, 0, 0],
        mode="write",
    )

    # Then replace it using bbox + resolution
    bbox = BBox3D.from_coords([5, 5, 2], [105, 105, 12], [4.0, 4.0, 40.0])

    layer = build_annotation_layer(
        path=path,
        bbox=bbox,
        resolution=[4.0, 4.0, 40.0],
        mode="replace",
    )

    assert isinstance(layer, VolumetricAnnotationLayer)
    assert layer.readonly is False

    # Verify the backend has the new bbox
    backend = layer.backend
    assert backend.index is not None
    expected_index = VolumetricIndex(bbox=bbox, resolution=Vec3D(4.0, 4.0, 40.0))
    assert backend.index.bbox == expected_index.bbox

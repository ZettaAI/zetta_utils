# pylint: disable=redefined-outer-name
import os
import shutil
import tempfile

import pytest

from zetta_utils import builder
from zetta_utils.geometry import Vec3D
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

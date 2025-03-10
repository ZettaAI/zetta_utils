# pylint: disable=redefined-outer-name
import os
import shutil
import tempfile

import pytest

from zetta_utils.geometry import Vec3D
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

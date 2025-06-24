# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
from __future__ import annotations

import os
import tempfile

import pytest

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.annotation import VolumetricAnnotationLayer
from zetta_utils.layer.volumetric.annotation.annotations import (  # Annotation,; PointAnnotation,
    LineAnnotation,
    PropertySpec,
)
from zetta_utils.layer.volumetric.annotation.backend import AnnotationLayerBackend


def create_backend(temp_dir, annotation_type="LINE", properties=None):
    backend_path = os.path.join(temp_dir, "annotation_backend")
    os.makedirs(backend_path, exist_ok=True)

    index = VolumetricIndex(
        resolution=Vec3D(1, 1, 1),
        bbox=BBox3D.from_slices((slice(0, 1000), slice(0, 1000), slice(0, 1000))),
    )

    backend = AnnotationLayerBackend(
        path=backend_path, annotation_type=annotation_type, index=index
    )
    if properties:
        backend.property_specs = properties
    return backend


def test_getitem_with_volumetric_index():
    with tempfile.TemporaryDirectory() as temp_dir:
        sample_annotations = [
            LineAnnotation(id=1, start=(10, 20, 30), end=(40, 50, 60)),
            LineAnnotation(id=2, start=(100, 200, 300), end=(400, 500, 600)),
        ]
        backend = create_backend(temp_dir)
        idx = VolumetricIndex(
            resolution=Vec3D(1, 1, 1),
            bbox=BBox3D.from_slices((slice(0, 1000), slice(0, 1000), slice(0, 1000))),
        )
        backend.write(idx=idx, data=sample_annotations)

        backend = create_backend(temp_dir)
        layer = VolumetricAnnotationLayer(
            backend=backend,
            index_resolution=Vec3D(1, 1, 1),
            allow_slice_rounding=False,
        )

        read_idx = VolumetricIndex(
            resolution=Vec3D(1, 1, 1),
            bbox=BBox3D.from_slices((slice(0, 500), slice(0, 500), slice(0, 500))),
        )

        result = layer[read_idx]

        assert len(result) == 2
        assert result[0] in sample_annotations
        assert result[1] in sample_annotations


def test_getitem_with_resolution_and_slices():
    with tempfile.TemporaryDirectory() as temp_dir:
        sample_annotations = [
            LineAnnotation(id=1, start=(10, 20, 30), end=(40, 50, 60)),
        ]
        backend = create_backend(temp_dir)
        idx = VolumetricIndex(
            resolution=Vec3D(1, 1, 1),
            bbox=BBox3D.from_slices((slice(0, 1000), slice(0, 1000), slice(0, 1000))),
        )
        backend.write(idx=idx, data=sample_annotations)

        backend = create_backend(temp_dir)
        layer = VolumetricAnnotationLayer(
            backend=backend,
            index_resolution=Vec3D(1, 1, 1),
            allow_slice_rounding=False,
        )

        result = layer[Vec3D(5, 5, 5), 0:500, 0:500, 0:500]

        assert len(result) == 1
        assert result[0].id == 1
        assert result[0].start == (2, 4, 6)
        assert result[0].end == (8, 10, 12)


def test_setitem_with_volumetric_index():
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = create_backend(temp_dir)
        layer = VolumetricAnnotationLayer(
            backend=backend,
            index_resolution=Vec3D(1, 1, 1),
            default_desired_resolution=Vec3D(2, 2, 2),
            allow_slice_rounding=False,
        )

        annotations = [
            LineAnnotation(id=1, start=(5, 10, 15), end=(20, 25, 30)),
        ]

        idx = VolumetricIndex(
            resolution=Vec3D(1, 1, 1),
            bbox=BBox3D.from_slices((slice(0, 500), slice(0, 500), slice(0, 500))),
        )

        layer[idx] = annotations

        read_idx = VolumetricIndex(
            resolution=Vec3D(1, 1, 1),
            bbox=BBox3D.from_slices((slice(0, 1000), slice(0, 1000), slice(0, 1000))),
        )

        backend = create_backend(temp_dir)
        result = backend.read(idx=read_idx)

        assert result == annotations


def test_setitem_with_resolution_and_slices():
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = create_backend(temp_dir)
        layer = VolumetricAnnotationLayer(
            backend=backend,
            index_resolution=Vec3D(1, 1, 1),
            allow_slice_rounding=False,
        )

        annotations = [
            LineAnnotation(start=(2, 4, 6), end=(8, 10, 12)),
        ]
        layer[Vec3D(5, 5, 5), 0:500, 0:500, 0:500] = annotations

        backend = create_backend(temp_dir)
        layer = VolumetricAnnotationLayer(
            backend=backend,
            index_resolution=Vec3D(1, 1, 1),
            allow_slice_rounding=False,
        )
        read_idx = VolumetricIndex(
            resolution=Vec3D(1, 1, 1),
            bbox=BBox3D.from_slices((slice(0, 1000), slice(0, 1000), slice(0, 1000))),
        )

        written_annotations = layer[read_idx]

        assert len(written_annotations) == 1
        assert written_annotations[0].start == (10, 20, 30)
        assert written_annotations[0].end == (40, 50, 60)


def test_setitem_with_properties():
    with tempfile.TemporaryDirectory(delete=False) as temp_dir:
        properties = (
            PropertySpec("score", "float32", "Score value in range [0,1]"),
            PropertySpec("score_pct", "uint8", "Int score in range [0,100]"),
        )
        backend = create_backend(temp_dir, "LINE", properties)
        layer = VolumetricAnnotationLayer(
            backend=backend,
            index_resolution=Vec3D(1, 1, 1),
            allow_slice_rounding=False,
        )

        annotations = [
            LineAnnotation(
                start=(2, 4, 6), end=(8, 10, 12), properties={"score": 0.42, "score_pct": 42}
            ),
        ]
        layer[Vec3D(5, 5, 5), 0:500, 0:500, 0:500] = annotations
        layer.backend.post_process()
        print(f"Wrote annotations to {temp_dir}")

        backend = create_backend(temp_dir)
        assert backend.property_specs == properties
        layer = VolumetricAnnotationLayer(
            backend=backend,
            index_resolution=Vec3D(1, 1, 1),
            allow_slice_rounding=False,
        )
        read_idx = VolumetricIndex(
            resolution=Vec3D(1, 1, 1),
            bbox=BBox3D.from_slices((slice(0, 1000), slice(0, 1000), slice(0, 1000))),
        )

        written_annotations = layer[read_idx]

        assert len(written_annotations) == 1
        assert written_annotations[0].start == (10, 20, 30)
        assert written_annotations[0].end == (40, 50, 60)
        assert written_annotations[0].properties["score"] == pytest.approx(0.42)
        assert written_annotations[0].properties["score_pct"] == 42


def test_with_changes():
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = create_backend(temp_dir)
        layer = VolumetricAnnotationLayer(
            backend=backend,
            index_resolution=Vec3D(1, 1, 1),
            default_desired_resolution=Vec3D(1, 1, 1),
            allow_slice_rounding=False,
        )

        new_layer = layer.with_changes(
            index_resolution=Vec3D(2, 2, 2),
            default_desired_resolution=Vec3D(4, 4, 4),
            allow_slice_rounding=True,
        )

        assert new_layer.index_resolution == Vec3D(2, 2, 2)
        assert new_layer.default_desired_resolution == Vec3D(4, 4, 4)
        assert new_layer.allow_slice_rounding is True
        assert new_layer.backend is backend

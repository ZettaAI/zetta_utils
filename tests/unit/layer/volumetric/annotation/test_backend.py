import os
import shutil

import pytest

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric.annotation import backend
from zetta_utils.layer.volumetric.annotation.backend import (
    AnnotationLayerBackend,
    LineAnnotation,
)
from zetta_utils.layer.volumetric.index import VolumetricIndex


def test_round_trip():
    temp_dir = os.path.expanduser("~/temp/test_precomp_anno")
    os.makedirs(temp_dir, exist_ok=True)
    file_dir = os.path.join(temp_dir, "round_trip")

    shutil.rmtree(file_dir, ignore_errors=True)  # start with a clean state

    lines = [
        LineAnnotation(id=1, start=(1640.0, 1308.0, 61.0), end=(1644.0, 1304.0, 57.0)),
        LineAnnotation(id=2, start=(1502.0, 1709.0, 589.0), end=(1498.0, 1701.0, 589.0)),
        LineAnnotation(id=3, start=(254.0, 68.0, 575.0), end=(258.0, 62.0, 575.0)),
        LineAnnotation(id=4, start=(1061.0, 657.0, 507.0), end=(1063.0, 653.0, 502.0)),
        LineAnnotation(id=5, start=(1298.0, 889.0, 315.0), end=(1295.0, 887.0, 314.0)),
    ]
    # Note: line 2 above, with the chunk_sizes below, will span 2 chunks, and so will
    # be written out to both of them.

    index = VolumetricIndex.from_coords([0, 0, 0], [2000, 2000, 600], Vec3D(10, 10, 40))

    chunk_sizes = [(2000, 2000, 600)]
    sf = AnnotationLayerBackend(
        path=file_dir, index=index, annotation_type="LINE", chunk_sizes=chunk_sizes
    )
    assert sf.chunk_sizes == [(2000, 2000, 600)]

    chunk_sizes = [(2000, 2000, 600), (1000, 1000, 600), (500, 500, 300)]
    sf = AnnotationLayerBackend(
        path=file_dir, index=index, annotation_type="LINE", chunk_sizes=chunk_sizes
    )
    os.makedirs(os.path.join(file_dir, "spatial0", "junkforcodecoverage"))
    sf.clear()
    sf.write_annotations([])  # (does nothing)
    sf.write_annotations(lines)
    sf.post_process()

    # Now create a *new* AnnotationLayer, given just the directory.
    sf = AnnotationLayerBackend(
        path=file_dir, index=index, annotation_type="LINE", chunk_sizes=chunk_sizes
    )
    assert sf.index == index
    assert sf.chunk_sizes == chunk_sizes

    lines_read = sf.read_all()
    assert len(lines_read) == len(lines)
    for line in lines:
        assert line in lines_read

    # Let's test with a bounds specified in *different* units than the file itself.
    roi = BBox3D.from_coords((510, 0, 300), (3000, 200, 1000), Vec3D(5, 50, 40))
    # With that index, and strict=False, we would get at least 3 lines (ids 3, 4, and 5).
    # And on this test, we'll get our coordinates in their original resolution.
    lines_read = sf.read_in_bounds(roi, strict=False)
    assert len(lines_read) == 3
    for line in lines_read:
        assert line in lines  # should match what was written exactly in this case
        assert line.id in [3, 4, 5]
        if line.id == 3:
            assert line.start == (254.0, 68.0, 575.0)
            assert line.end == (258.0, 62.0, 575.0)
    # But with strict=True, we should get only 2 lines (ids 4 and 5).
    # And also, in this test, we'll ask for the coordinates in nm.
    lines_read = sf.read_in_bounds(roi, strict=True, annotation_resolution=Vec3D(1, 1, 1))
    assert len(lines_read) == 2
    for line in lines_read:
        assert line.id in [4, 5]
        if line.id == 3:
            assert line.start == (254.0 * 10, 68.0 * 10, 575.0 * 40)
            assert line.end == (258.0 * 10, 62.0 * 10, 575.0 * 40)

    # Test replacing only the two lines in that bounds.
    new_lines = [
        LineAnnotation(id=104, start=(1061.5, 657.0, 507.0), end=(1062.5, 653.0, 502.0)),
        LineAnnotation(id=105, start=(1298.5, 889.0, 315.0), end=(1294.5, 887.0, 314.0)),
    ]
    sf.write_annotations(new_lines, clearing_bbox=roi)
    lines_read = sf.read_in_bounds(roi, strict=False)
    assert len(lines_read) == 3
    for line in lines_read:
        assert line.id in [3, 104, 105]
        if line.id == 104:
            assert line.start == (1061.5, 657.0, 507.0)
            assert line.end == (1062.5, 653.0, 502.0)

    # Above is typical usage.  Below, we do some odd things
    # to trigger other code paths we want to test.
    lines_read = sf.read_all(-1, False)  # allow duplicates
    assert len(lines_read) == len(lines) + 1

    # shutil.rmtree(os.path.join(file_dir, "spatial0"))
    # entries = backend.subdivide([], sf.index, sf.chunk_sizes, file_dir)
    # assert len(entries) == 3
    # backend.read_data(file_dir, entries[0])

    shutil.rmtree(file_dir)  # clean up when done


def test_resolution_changes():
    temp_dir = os.path.expanduser("~/temp/test_precomp_anno")
    os.makedirs(temp_dir, exist_ok=True)
    file_dir = os.path.join(temp_dir, "resolution_changes")

    # file resolution: 20, 20, 40
    resolution = Vec3D(20, 20, 40)
    voxel_offset = [0, 0, 0]
    dataset_size = [2000, 2000, 600]
    end_coord = tuple(a + b for a, b in zip(voxel_offset, dataset_size))
    index = VolumetricIndex.from_coords(voxel_offset, end_coord, resolution)
    sf = AnnotationLayerBackend(path=file_dir, index=index, annotation_type="LINE")
    sf.clear()

    # writing with voxel size 10, 10, 80
    lines = [LineAnnotation(id=1, start=(100, 500, 50), end=(200, 600, 60))]
    sf.write_annotations(lines, Vec3D(10, 10, 80))

    # pull those back out at file native resolution, i.e. (20, 20, 40)
    lines_read = sf.read_all()
    assert len(lines_read) == 1
    assert lines_read[0].start == (100 * 10 / 20, 500 * 10 / 20, 50 * 80 / 40)
    assert lines_read[0].end == (200 * 10 / 20, 600 * 10 / 20, 60 * 80 / 40)

    # pull those back out at resolution (5, 5, 20)
    lines_read = sf.read_all(annotation_resolution=Vec3D(5, 5, 20))
    assert len(lines_read) == 1
    assert lines_read[0].start == (100 * 10 / 5, 500 * 10 / 5, 50 * 80 / 20)
    assert lines_read[0].end == (200 * 10 / 5, 600 * 10 / 5, 60 * 80 / 20)


def test_single_level():
    temp_dir = os.path.expanduser("~/temp/test_precomp_anno")
    os.makedirs(temp_dir, exist_ok=True)
    file_dir = os.path.join(temp_dir, "single_level")
    shutil.rmtree(file_dir, ignore_errors=True)

    lines = [
        LineAnnotation(id=1, start=(1640.0, 1308.0, 61.0), end=(1644.0, 1304.0, 57.0)),
        LineAnnotation(id=2, start=(1502.0, 1709.0, 589.0), end=(1498.0, 1701.0, 589.0)),
        LineAnnotation(id=3, start=(254.0, 68.0, 575.0), end=(258.0, 62.0, 575.0)),
        LineAnnotation(id=4, start=(1061.0, 657.0, 507.0), end=(1063.0, 653.0, 502.0)),
        LineAnnotation(id=5, start=(1298.0, 889.0, 315.0), end=(1295.0, 887.0, 314.0)),
    ]
    # Note: line 2 above, with the chunk_sizes below, will span 2 chunks, and so will
    # be written out to both of them.

    index = VolumetricIndex.from_coords([0, 0, 0], [2000, 2000, 600], Vec3D(10, 10, 40))

    sf = AnnotationLayerBackend(path=file_dir, index=index, annotation_type="LINE")
    assert sf.chunk_sizes == [(2000, 2000, 600)]

    chunk_sizes = [[500, 500, 300]]
    sf = AnnotationLayerBackend(
        path=file_dir, index=index, annotation_type="LINE", chunk_sizes=chunk_sizes
    )
    os.makedirs(os.path.join(file_dir, "spatial0", "junkforcodecoverage"))
    sf.clear()
    sf.write_annotations([])  # (does nothing)
    sf.write_annotations(lines)
    sf.post_process()

    chunk_path = os.path.join(file_dir, "spatial0", "2_1_1")
    assert backend.count_lines_in_file(chunk_path) == 2


def test_edge_cases():
    with pytest.raises(ValueError):
        backend.path_join()

    assert backend.read_info("/dev/null") == (None, None, None, None, None, None, None)

    assert backend.path_join("gs://foo/", "bar") == "gs://foo/bar"

    index = VolumetricIndex.from_coords((0, 0, 0), (10, 10, 10), (1, 1, 1))
    for path in ["/dev/null", "/dev/null/subdir"]:
        assert not AnnotationLayerBackend(path=path, index=index, annotation_type="LINE").exists()


def test_getter_methods():
    # Create a test backend with a known index
    index = VolumetricIndex.from_coords([100, 200, 300], [500, 600, 700], Vec3D(10, 20, 40))
    chunk_sizes = [(200, 200, 200), (100, 100, 100)]
    backend_instance = AnnotationLayerBackend(
        path="/tmp/test", index=index, annotation_type="LINE", chunk_sizes=chunk_sizes
    )

    # Test get_voxel_offset with different resolutions
    voxel_offset_native = backend_instance.get_voxel_offset(Vec3D(10, 20, 40))
    assert voxel_offset_native == Vec3D(100, 200, 300)

    voxel_offset_half = backend_instance.get_voxel_offset(Vec3D(5, 10, 20))
    assert voxel_offset_half == Vec3D(50, 100, 150)

    # Test get_chunk_size with different resolutions (uses chunk_sizes[0] = (200, 200, 200))
    chunk_size_native = backend_instance.get_chunk_size(Vec3D(10, 20, 40))
    assert chunk_size_native == Vec3D(200, 200, 200)

    chunk_size_double = backend_instance.get_chunk_size(Vec3D(20, 40, 80))
    assert chunk_size_double == Vec3D(400, 400, 400)

    # Test get_dataset_size with different resolutions (index.shape = [400, 400, 400])
    dataset_size_native = backend_instance.get_dataset_size(Vec3D(10, 20, 40))
    assert dataset_size_native == Vec3D(400, 400, 400)

    dataset_size_quarter = backend_instance.get_dataset_size(Vec3D(2.5, 5, 10))
    assert dataset_size_quarter == Vec3D(100, 100, 100)

    # Test get_bounds with different resolutions
    bounds_native = backend_instance.get_bounds(Vec3D(10, 20, 40))
    expected_bounds = index * Vec3D(10, 20, 40) / Vec3D(10, 20, 40)
    assert bounds_native == expected_bounds

    bounds_half = backend_instance.get_bounds(Vec3D(5, 10, 20))
    expected_bounds_half = index * Vec3D(5, 10, 20) / Vec3D(10, 20, 40)
    assert bounds_half == expected_bounds_half

    # Test pformat
    pformat_result = backend_instance.pformat()
    assert isinstance(pformat_result, str)
    assert len(pformat_result) > 0


if __name__ == "__main__":
    test_round_trip()
    test_edge_cases()

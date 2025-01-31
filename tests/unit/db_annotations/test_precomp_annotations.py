import os
import shutil

import pytest

from zetta_utils.db_annotations import precomp_annotations
from zetta_utils.db_annotations.precomp_annotations import (
    AnnotationLayer,
    LineAnnotation,
)
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric.index import VolumetricIndex


def test_round_trip():
    temp_dir = os.path.expanduser("~/temp/test_precomp_anno")
    os.makedirs(temp_dir, exist_ok=True)
    file_dir = os.path.join(temp_dir, "round_trip")

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

    sf = AnnotationLayer(file_dir, index)
    assert sf.chunk_sizes == [(2000, 2000, 600)]

    chunk_sizes = [[2000, 2000, 600], [1000, 1000, 600], [500, 500, 300]]
    sf = AnnotationLayer(file_dir, index, chunk_sizes)
    os.makedirs(os.path.join(file_dir, "spatial0", "junkforcodecoverage"))
    sf.clear()
    sf.write_annotations([])  # (does nothing)
    sf.write_annotations(lines)
    sf.post_process()

    # Now create a *new* AnnotationLayer, given just the directory.
    sf = AnnotationLayer(file_dir)
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

    shutil.rmtree(os.path.join(file_dir, "spatial0"))
    entries = precomp_annotations.subdivide([], sf.index, sf.chunk_sizes, file_dir)
    assert len(entries) == 3
    precomp_annotations.read_data(file_dir, entries[0])

    shutil.rmtree(file_dir)  # clean up when done


def test_resolution_changes():
    temp_dir = os.path.expanduser("~/temp/test_precomp_anno")
    os.makedirs(temp_dir, exist_ok=True)
    file_dir = os.path.join(temp_dir, "resolution_changes")

    # file resolution: 20, 20, 40
    index = VolumetricIndex.from_coords([0, 0, 0], [2000, 2000, 600], Vec3D(20, 20, 40))
    sf = AnnotationLayer(file_dir, index)
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

    sf = AnnotationLayer(file_dir, index)
    assert sf.chunk_sizes == [(2000, 2000, 600)]

    chunk_sizes = [[500, 500, 300]]
    sf = AnnotationLayer(file_dir, index, chunk_sizes)
    os.makedirs(os.path.join(file_dir, "spatial0", "junkforcodecoverage"))
    sf.clear()
    sf.write_annotations([])  # (does nothing)
    sf.write_annotations(lines)
    sf.post_process()

    chunk_path = os.path.join(file_dir, "spatial0", "2_1_1")
    assert precomp_annotations.count_lines_in_file(chunk_path) == 2

    with pytest.raises(ValueError):
        out_of_bounds_lines = [
            LineAnnotation(id=1, start=(1640.0, 1308.0, 61.0), end=(1644.0, 1304.0, 57.0)),
            LineAnnotation(id=666, start=(-100, 0, 0), end=(50, 50, 50)),
        ]
        roi = BBox3D.from_coords((25, 25, 25), (250, 250, 250), resolution=(10, 10, 40))
        sf.write_annotations(out_of_bounds_lines, clearing_bbox=roi)


def test_edge_cases():
    with pytest.raises(ValueError):
        precomp_annotations.path_join()

    # pylint: disable=use-implicit-booleaness-not-comparison
    assert precomp_annotations.read_lines("/dev/null") == []

    assert precomp_annotations.read_info("/dev/null") == (None, None, None, None)

    assert precomp_annotations.path_join("gs://foo/", "bar") == "gs://foo/bar"

    index = VolumetricIndex.from_coords((0, 0, 0), (10, 10, 10), (1, 1, 1))
    assert not AnnotationLayer("/dev/null", index).exists()
    assert not AnnotationLayer("/dev/null/subdir", index).exists()


if __name__ == "__main__":
    test_round_trip()
    test_edge_cases()

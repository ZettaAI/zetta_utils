import os
import shutil

import pytest

from zetta_utils.db_annotations import precomp_annotations
from zetta_utils.db_annotations.precomp_annotations import (
    AnnotationLayer,
    LineAnnotation,
)
from zetta_utils.geometry.vec import Vec3D
from zetta_utils.layer.volumetric.index import VolumetricIndex


def test_round_trip():
    temp_dir = os.path.expanduser("~/temp/test_precomp_anno")
    os.makedirs(temp_dir, exist_ok=True)
    file_dir = os.path.join(temp_dir, "round_trip")

    lines = [
        LineAnnotation(line_id=1, start=(1640.0, 1308.0, 61.0), end=(1644.0, 1304.0, 57.0)),
        LineAnnotation(line_id=2, start=(1502.0, 1709.0, 589.0), end=(1498.0, 1701.0, 589.0)),
        LineAnnotation(line_id=3, start=(254.0, 68.0, 575.0), end=(258.0, 62.0, 575.0)),
        LineAnnotation(line_id=4, start=(1061.0, 657.0, 507.0), end=(1063.0, 653.0, 502.0)),
        LineAnnotation(line_id=5, start=(1298.0, 889.0, 315.0), end=(1295.0, 887.0, 314.0)),
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

    # Above is typical usage.  Below, we do some odd things
    # to trigger other code paths we want to test.
    lines_read = sf.read_all(-1, False)  # allow duplicates
    assert len(lines_read) == len(lines) + 1

    shutil.rmtree(os.path.join(file_dir, "spatial0"))
    entries = precomp_annotations.subdivide([], sf.index, sf.chunk_sizes, file_dir)
    assert len(entries) == 3
    precomp_annotations.read_data(file_dir, entries[0])

    shutil.rmtree(file_dir)  # clean up when done


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

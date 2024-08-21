import os
import shutil
from random import randrange

import pytest

from zetta_utils.db_annotations import precomp_annotations
from zetta_utils.db_annotations.precomp_annotations import LineAnnotation, SpatialFile
from zetta_utils.geometry.vec import Vec3D
from zetta_utils.layer.volumetric.index import VolumetricIndex


def test_round_trip():
    temp_dir = os.path.expanduser("~/temp/test_precomp_anno")
    os.makedirs(temp_dir, exist_ok=True)
    file_dir = os.path.join(temp_dir, "round_trip")

    lines = []
    line_id = 1
    for _ in range(5):
        x = randrange(0, 2000)
        y = randrange(0, 2000)
        z = randrange(0, 600)
        dx = randrange(-10, 10)
        dy = randrange(-10, 10)
        dz = randrange(-5, 5)
        lines.append(LineAnnotation(line_id, (x, y, z), (x + dx, y + dy, z + dz)))
        line_id += 1

    index = VolumetricIndex.from_coords([0, 0, 0], [2000, 2000, 600], Vec3D(10, 10, 40))

    sf = SpatialFile(file_dir, index)
    assert sf.chunk_sizes == [(2000, 2000, 600)]

    chunk_sizes = [[2000, 2000, 600], [1000, 1000, 600], [500, 500, 300]]
    sf = SpatialFile(file_dir, index, chunk_sizes)
    sf.clear()
    sf.write_annotations([])  # (does nothing)
    sf.write_annotations(lines)
    sf.post_process()

    # Now create a *new* SpatialFile, given just the directory.
    sf = SpatialFile(file_dir)
    assert sf.index == index
    assert sf.chunk_sizes == chunk_sizes

    lines_read = sf.read_all()
    assert len(lines_read) == len(lines)
    for line in lines:
        assert line in lines_read

    # Above is typical usage.  Below, we do some odd things
    # to trigger other code paths we want to test.
    shutil.rmtree(os.path.join(file_dir, "spatial0"))
    entries = precomp_annotations.subdivide([], sf.index, sf.chunk_sizes, file_dir)
    assert len(entries) == 3
    precomp_annotations.read_data(file_dir, entries[0])


def test_edge_cases():
    with pytest.raises(ValueError):
        precomp_annotations.path_join()

    # pylint: disable=use-implicit-booleaness-not-comparison
    assert precomp_annotations.read_lines("/dev/null") == []

    assert precomp_annotations.path_join("gs://foo/", "bar") == "gs://foo/bar"


if __name__ == "__main__":
    test_round_trip()
    test_edge_cases()

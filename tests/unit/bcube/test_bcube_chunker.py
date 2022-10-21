# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
from zetta_utils.bcube import BoundingCube, BcubeStrider


def test_bcube_rounding(mocker):
    chunker = BcubeStrider(
        bcube=BoundingCube.from_coords(
            start_coord=(0, 0, 0), end_coord=(1, 1, 4), resolution=(1, 1, 1)
        ),
        chunk_size=(1, 1, 3),
        step_size=(1, 1, 3),
        resolution=(1, 1, 1),
    )
    assert chunker.num_chunks == 1
    assert chunker.step_limits == (1, 1, 1)


def test_bcube_chunker_len(mocker):
    chunker = BcubeStrider(
        bcube=BoundingCube.from_coords(
            start_coord=(0, 0, 0), end_coord=(1, 2, 3), resolution=(1, 1, 1)
        ),
        chunk_size=(1, 1, 1),
        resolution=(1, 1, 1),
        step_size=(1, 1, 1),
    )
    assert chunker.num_chunks == 6


def test_bcube_chunker_get_nth(mocker):
    chunker = BcubeStrider(
        bcube=BoundingCube.from_coords(
            start_coord=(0, 0, 0), end_coord=(1, 2, 3), resolution=(1, 1, 1)
        ),
        chunk_size=(1, 1, 1),
        resolution=(1, 1, 1),
        step_size=(1, 1, 1),
    )
    assert chunker.get_nth_chunk_bcube(0) == BoundingCube.from_slices(
        (slice(0, 1), slice(0, 1), slice(0, 1))
    )
    assert chunker.get_nth_chunk_bcube(1) == BoundingCube.from_slices(
        (slice(0, 1), slice(1, 2), slice(0, 1))
    )
    assert chunker.get_nth_chunk_bcube(4) == BoundingCube.from_slices(
        (slice(0, 1), slice(0, 1), slice(2, 3))
    )
    # assert vsi(10) == (None, slice(10, 11), slice(200, 202), slice(3100, 3103))


def test_bcube_chunker_get_nth_res(mocker):
    chunker = BcubeStrider(
        bcube=BoundingCube.from_coords(
            start_coord=(0, 0, 0), end_coord=(1, 2, 3), resolution=(1, 1, 1)
        ),
        chunk_size=(1, 1, 1),
        step_size=(1, 1, 1),
        resolution=(1, 2, 1),
    )
    assert chunker.num_chunks == 3
    assert chunker.get_nth_chunk_bcube(0) == BoundingCube.from_slices(
        (slice(0, 1), slice(0, 2), slice(0, 1))
    )
    assert chunker.get_nth_chunk_bcube(1) == BoundingCube.from_slices(
        (slice(0, 1), slice(0, 2), slice(1, 2))
    )


def test_bcube_chunker_get_all_chunks(mocker):
    chunker = BcubeStrider(
        bcube=BoundingCube.from_coords(
            start_coord=(0, 0, 0), end_coord=(1, 1, 2), resolution=(1, 1, 1)
        ),
        chunk_size=(1, 1, 1),
        step_size=(1, 1, 1),
        resolution=(1, 1, 1),
    )
    assert chunker.get_all_chunk_bcubes() == [
        BoundingCube.from_slices((slice(0, 1), slice(0, 1), slice(0, 1))),
        BoundingCube.from_slices((slice(0, 1), slice(0, 1), slice(1, 2))),
    ]

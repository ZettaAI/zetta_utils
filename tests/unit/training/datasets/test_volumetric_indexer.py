# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
from zetta_utils.training.datasets.sample_indexers import VolumetricStepIndexer
from zetta_utils.bcube import BoundingCube


def test_len(mocker):
    vsi = VolumetricStepIndexer(
        bcube=BoundingCube.from_coords(
            start_coord=(0, 0, 0), end_coord=(1, 2, 3), resolution=(1, 1, 1)
        ),
        chunk_size=(1, 1, 1),
        resolution=(1, 1, 1),
        step_size=(1, 1, 1),
    )
    assert len(vsi) == 6


def test_call_default(mocker):
    vsi = VolumetricStepIndexer(
        bcube=BoundingCube.from_coords(
            start_coord=(0, 0, 0), end_coord=(1, 2, 3), resolution=(1, 1, 1)
        ),
        chunk_size=(1, 1, 1),
        resolution=(1, 1, 1),
        step_size=(1, 1, 1),
    )
    assert vsi(0) == (None, slice(0, 1), slice(0, 1), slice(0, 1))


def test_call_index_res(mocker):
    vsi = VolumetricStepIndexer(
        bcube=BoundingCube.from_coords(
            start_coord=(0, 0, 0), end_coord=(1, 2, 3), resolution=(1, 1, 1)
        ),
        chunk_size=(1, 1, 1),
        resolution=(1, 1, 1),
        step_size=(1, 1, 1),
        index_resolution=(0.5, 0.5, 0.5),
    )
    assert vsi(0) == (None, slice(0, 2), slice(0, 2), slice(0, 2))


def test_call_desired_res(mocker):
    vsi = VolumetricStepIndexer(
        bcube=BoundingCube.from_coords(
            start_coord=(0, 0, 0), end_coord=(1, 2, 3), resolution=(1, 1, 1)
        ),
        chunk_size=(1, 1, 1),
        resolution=(1, 1, 1),
        step_size=(1, 1, 1),
        desired_resolution=(0.5, 0.5, 0.5),
    )
    assert vsi(0) == ((0.5, 0.5, 0.5), slice(0, 1), slice(0, 1), slice(0, 1))

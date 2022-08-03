# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
from zetta_utils.training.datasets.sample_indexers import VolumetricStepIndexer
from zetta_utils.bbox import BoundingCube


def test_volumetric_step_indexer_constructor(mocker):
    vsi = VolumetricStepIndexer(
        bcube=BoundingCube.from_coords(
            start_coord=(10, 20, 30), end_coord=(11, 22, 33), resolution=(1, 10, 100)
        ),
        sample_size=(0.5, 1, 1.5),
        sample_size_resolution=(2, 2, 2),
        step_size=(2, 2, 10),
        step_size_resolution=(1, 1, 10),
        index_resolution=(1, 1, 1),
    )
    assert vsi.sample_size_in_unit == (1, 2, 3)
    assert vsi.step_size_in_unit == (2, 2, 100)
    assert vsi.step_limits == (1, 10, 3)


def test_volumetric_step_indexer_len(mocker):
    vsi = VolumetricStepIndexer(
        bcube=BoundingCube.from_coords(
            start_coord=(10, 20, 30), end_coord=(11, 22, 33), resolution=(1, 10, 100)
        ),
        sample_size=(0.5, 1, 1.5),
        sample_size_resolution=(2, 2, 2),
        step_size=(2, 2, 10),
        step_size_resolution=(1, 1, 10),
        index_resolution=(1, 1, 1),
    )
    assert len(vsi) == 30


def test_volumetric_step_indexer_getitem(mocker):
    vsi = VolumetricStepIndexer(
        bcube=BoundingCube.from_coords(
            start_coord=(10, 20, 30), end_coord=(11, 22, 33), resolution=(1, 10, 100)
        ),
        sample_size=(0.5, 1, 1.5),
        sample_size_resolution=(2, 2, 2),
        step_size=(2, 2, 10),
        step_size_resolution=(1, 1, 10),
        index_resolution=(1, 1, 1),
    )
    assert vsi(0) == (None, slice(10, 11), slice(200, 202), slice(3000, 3003))
    assert vsi(1) == (None, slice(10, 11), slice(202, 204), slice(3000, 3003))
    assert vsi(10) == (None, slice(10, 11), slice(200, 202), slice(3100, 3103))

# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
import numpy as np
import pytest

from zetta_utils.bcube import BoundingCube
from zetta_utils.layer.volumetric import VolumetricIndex, build_volumetric_layer


def test_build_exc(mocker):
    with pytest.raises(ValueError):
        build_volumetric_layer(backend=mocker.MagicMock(), data_resolution=(2, 1, 1))


def test_data_resolution_read_interp(mocker):
    backend = mocker.MagicMock()
    backend.read = mocker.MagicMock(return_value=np.ones((2, 2, 2, 2)) * 2)

    layer = build_volumetric_layer(
        backend,
        data_resolution=(2, 2, 2),
        default_desired_resolution=(4, 4, 4),
        interpolation_mode="field",
        index_resolution=(3, 3, 3),
    )

    read_data = layer[0:1, 0:1, 0:1]

    backend.read.assert_called_with(
        idx=VolumetricIndex(
            resolution=(2, 2, 2),
            bcube=BoundingCube.from_slices((slice(0, 3), slice(0, 3), slice(0, 3))),
        )
    )
    np.testing.assert_array_equal(
        read_data,
        np.ones((2, 1, 1, 1)),
    )


def test_data_resolution_write_interp(mocker):
    backend = mocker.MagicMock()
    backend.write = mocker.MagicMock()

    layer = build_volumetric_layer(
        backend,
        data_resolution=(2, 2, 2),
        default_desired_resolution=(4, 4, 4),
        interpolation_mode="field",
        index_resolution=(3, 3, 3),
    )

    idx = VolumetricIndex(
        resolution=(2, 2, 2),
        bcube=BoundingCube.from_slices((slice(0, 3), slice(0, 3), slice(0, 3))),
    )

    layer[0:1, 0:1, 0:1] = np.ones((2, 1, 1, 1))
    assert backend.write.call_args.kwargs["idx"] == idx
    np.testing.assert_array_equal(
        backend.write.call_args.kwargs["value"],
        np.ones((2, 2, 2, 2)) * 2,
    )

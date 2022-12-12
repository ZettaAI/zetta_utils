# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
import pytest
import torch

from zetta_utils.bcube import BoundingCube
from zetta_utils.layer.volumetric import VolumetricIndex, build_volumetric_layer


def test_build_exc(mocker):
    with pytest.raises(ValueError):
        build_volumetric_layer(backend=mocker.MagicMock(), data_resolution=(2, 1, 1))


def test_data_resolution_read_interp(mocker):
    backend = mocker.MagicMock()
    backend.read = mocker.MagicMock(return_value=torch.ones((2, 2, 2, 2)) * 2)

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
    assert torch.equal(
        read_data,
        torch.ones((2, 1, 1, 1)),
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

    layer[0:1, 0:1, 0:1] = torch.ones((2, 1, 1, 1))
    assert backend.write.call_args.kwargs["idx"] == idx
    assert torch.equal(
        backend.write.call_args.kwargs["data"],
        torch.ones((2, 2, 2, 2)) * 2,
    )


def test_write_scalar(mocker):
    backend = mocker.MagicMock()
    backend.write = mocker.MagicMock()

    layer = build_volumetric_layer(
        backend,
        default_desired_resolution=(1, 1, 1),
        index_resolution=(1, 1, 1),
    )

    layer[0:1, 0:1, 0:1] = 1.0
    assert torch.equal(
        backend.write.call_args.kwargs["data"],
        torch.Tensor([1]),
    )


def test_write_scalar_with_processor(mocker):
    backend = mocker.MagicMock()
    backend.write = mocker.MagicMock()

    layer = build_volumetric_layer(
        backend,
        default_desired_resolution=(1, 1, 1),
        index_resolution=(1, 1, 1),
        write_preprocs=[lambda data: data + 1],
    )

    layer[0:1, 0:1, 0:1] = 1.0
    assert torch.equal(
        backend.write.call_args.kwargs["data"],
        torch.Tensor([2]),
    )

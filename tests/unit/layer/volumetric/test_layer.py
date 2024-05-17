# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
from __future__ import annotations

import numpy as np
import pytest

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricIndexTranslator,
    build_volumetric_layer,
)

from ...helpers import assert_array_equal


def test_data_resolution_read_interp(mocker):
    backend = mocker.MagicMock()
    backend.read = mocker.MagicMock(return_value=np.ones((2, 2, 2, 2)) * 2)

    layer = build_volumetric_layer(
        backend,
        data_resolution=Vec3D(2, 2, 2),
        default_desired_resolution=Vec3D(4, 4, 4),
        interpolation_mode="field",
        index_resolution=Vec3D(3, 3, 3),
    )

    read_data = layer[0:1, 0:1, 0:1]

    backend.read.assert_called_with(
        idx=VolumetricIndex(
            resolution=Vec3D(2, 2, 2),
            bbox=BBox3D.from_slices((slice(0, 3), slice(0, 3), slice(0, 3))),
        )
    )
    assert_array_equal(
        read_data,
        np.ones((2, 1, 1, 1)),
    )


def test_data_resolution_interp_exc(mocker):
    with pytest.raises(ValueError):
        build_volumetric_layer(backend=mocker.MagicMock(), data_resolution=Vec3D(2, 1, 1))


def test_data_resolution_write_interp(mocker):
    backend = mocker.MagicMock()
    backend.write = mocker.MagicMock()

    layer = build_volumetric_layer(
        backend,
        data_resolution=Vec3D(2, 2, 2),
        index_resolution=Vec3D(1, 1, 1),
        default_desired_resolution=Vec3D(4, 4, 4),
        interpolation_mode="field",
    )

    idx = VolumetricIndex(
        resolution=Vec3D(2, 2, 2),
        bbox=BBox3D.from_slices((slice(0, 4), slice(0, 4), slice(0, 4))),
    )

    layer[0:4, 0:4, 0:4] = np.ones((2, 1, 1, 1))
    assert backend.write.call_args.kwargs["idx"] == idx
    assert_array_equal(
        backend.write.call_args.kwargs["data"],
        np.ones((2, 2, 2, 2)) * 2,
    )


def test_write_scalar(mocker):
    backend = mocker.MagicMock()
    backend.write = mocker.MagicMock()

    layer = build_volumetric_layer(
        backend,
        default_desired_resolution=Vec3D(1, 1, 1),
        index_resolution=Vec3D(1, 1, 1),
    )

    layer[0:1, 0:1, 0:1] = 1.0
    assert_array_equal(
        backend.write.call_args.kwargs["data"],
        np.array([1]),
    )


def test_write_scalar_with_processor(mocker):
    backend = mocker.MagicMock()
    backend.write = mocker.MagicMock()

    layer = build_volumetric_layer(
        backend,
        default_desired_resolution=Vec3D(1, 1, 1),
        index_resolution=Vec3D(1, 1, 1),
        write_procs=[lambda data: data + 1],
    )

    layer[0:1, 0:1, 0:1] = 1.0
    assert_array_equal(
        backend.write.call_args.kwargs["data"],
        np.array([2]),
    )


def test_read_write_with_idx_processor(mocker):
    backend = mocker.MagicMock()
    backend.write = mocker.MagicMock()
    backend.read = mocker.MagicMock(return_value=np.ones((2, 1, 1, 1)))

    layer = build_volumetric_layer(
        backend,
        default_desired_resolution=Vec3D(2, 2, 2),
        index_resolution=Vec3D(2, 2, 2),
        index_procs=[VolumetricIndexTranslator(offset=Vec3D(2, 4, 8), resolution=Vec3D(1, 1, 1))],
        write_procs=[lambda data: data + 1],
        read_procs=[lambda data: data - 1],
    )

    expected_idx = VolumetricIndex(
        resolution=Vec3D(2, 2, 2),
        bbox=BBox3D.from_slices((slice(2, 4), slice(4, 6), slice(8, 10))),
    )
    layer[0:1, 0:1, 0:1] = 1.0
    assert_array_equal(
        backend.write.call_args.kwargs["data"],
        np.array([2]),
    )
    assert backend.write.call_args.kwargs["idx"] == expected_idx

    data_read = layer[0:1, 0:1, 0:1]
    assert_array_equal(data_read, np.zeros((2, 1, 1, 1)))
    assert backend.write.call_args.kwargs["idx"] == expected_idx


def test_write_readonly_exc(mocker):
    backend = mocker.MagicMock()

    layer = build_volumetric_layer(
        backend,
        readonly=True,
        index_resolution=Vec3D(1, 1, 1),
        default_desired_resolution=Vec3D(4, 4, 4),
    )

    with pytest.raises(IOError):
        layer[0:1, 0:1, 0:1] = 1

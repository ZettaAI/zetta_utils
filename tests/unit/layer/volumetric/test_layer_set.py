import numpy as np

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, build_volumetric_layer_set


def test_read(mocker):
    layer_a = mocker.MagicMock()
    layer_a.read_with_procs = mocker.MagicMock(return_value=np.array([1]))
    layer_b = mocker.MagicMock()
    layer_b.read_with_procs = mocker.MagicMock(return_value=np.array([2]))
    layer_set = build_volumetric_layer_set(layers={"a": layer_a, "b": layer_b})
    idx = VolumetricIndex(bbox=BBox3D(bounds=((0, 1), (0, 1), (0, 1))), resolution=Vec3D(1, 1, 1))
    result = layer_set[idx]
    assert result == {"a": np.array([1]), "b": np.array([2])}
    layer_a.read_with_procs.assert_called_with(idx)
    layer_b.read_with_procs.assert_called_with(idx)


def test_write(mocker):
    layer_a = mocker.MagicMock()
    layer_a.write_with_procs = mocker.MagicMock()
    layer_b = mocker.MagicMock()
    layer_b.write_with_procs = mocker.MagicMock()
    layer_set = build_volumetric_layer_set(layers={"a": layer_a, "b": layer_b})
    idx = VolumetricIndex(bbox=BBox3D(bounds=((0, 1), (0, 1), (0, 1))), resolution=Vec3D(1, 1, 1))
    layer_set[idx] = {"a": 1, "b": 2}
    layer_a.write_with_procs.assert_called_with(idx, 1)
    layer_b.write_with_procs.assert_called_with(idx, 2)

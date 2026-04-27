import numpy as np

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, build_volumetric_layer_set
from zetta_utils.layer.volumetric.layer import VolumetricLayer


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


def test_with_changes_namespaces_name_per_layer(mocker):
    backend_a = mocker.MagicMock()
    backend_b = mocker.MagicMock()
    new_backend_a = mocker.MagicMock()
    new_backend_b = mocker.MagicMock()
    backend_a.with_changes = mocker.MagicMock(return_value=new_backend_a)
    backend_b.with_changes = mocker.MagicMock(return_value=new_backend_b)
    layer_a = VolumetricLayer(backend=backend_a)
    layer_b = VolumetricLayer(backend=backend_b)
    layer_set = build_volumetric_layer_set(layers={"a": layer_a, "b": layer_b})

    result = layer_set.backend.with_changes(name="parent", allow_cache=True)

    backend_a.with_changes.assert_called_once_with(name="parent/a", allow_cache=True)
    backend_b.with_changes.assert_called_once_with(name="parent/b", allow_cache=True)
    assert result.layers["a"].backend is new_backend_a
    assert result.layers["b"].backend is new_backend_b


def test_with_changes_propagates_kwargs_without_name(mocker):
    backend_a = mocker.MagicMock()
    backend_b = mocker.MagicMock()
    backend_a.with_changes = mocker.MagicMock(return_value=mocker.MagicMock())
    backend_b.with_changes = mocker.MagicMock(return_value=mocker.MagicMock())
    layer_a = VolumetricLayer(backend=backend_a)
    layer_b = VolumetricLayer(backend=backend_b)
    layer_set = build_volumetric_layer_set(layers={"a": layer_a, "b": layer_b})

    layer_set.backend.with_changes(allow_cache=True)

    backend_a.with_changes.assert_called_once_with(allow_cache=True)
    backend_b.with_changes.assert_called_once_with(allow_cache=True)
    assert "name" not in backend_a.with_changes.call_args.kwargs
    assert "name" not in backend_b.with_changes.call_args.kwargs


def test_delete(mocker):
    layer_a = mocker.MagicMock()
    layer_a.delete = mocker.MagicMock()
    layer_b = mocker.MagicMock()
    layer_b.delete = mocker.MagicMock()
    layer_set = build_volumetric_layer_set(layers={"a": layer_a, "b": layer_b})
    layer_set.delete()
    layer_a.delete.assert_called_once()
    layer_b.delete.assert_called_once()

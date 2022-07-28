# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
import pytest

from zetta_utils.io.layers import build_cv_layer, build_layer_set


def test_cv_layer_exc(mocker):
    mocker.patch("zetta_utils.io.backends.CVBackend")
    mocker.patch("zetta_utils.io.indexes.volumetric.VolumetricIndexConverter")

    with pytest.raises(ValueError):
        build_cv_layer(path="path", data_resolution=(1, 1, 1))


def test_cv_layer(mocker):
    kwargs = {
        "path": "path",
        "cv_kwargs": {"a": "b"},
        "default_desired_resolution": (1, 1, 1),
        "index_resolution": (2, 2, 2),
        "readonly": True,
        "data_resolution": None,
        "interpolation_mode": None,
        "allow_shape_rounding": False,
        "index_adjs": [mocker.Mock()],
        "read_postprocs": [mocker.Mock()],
        "write_preprocs": [mocker.Mock()],
    }

    cvl_x0 = build_cv_layer(**kwargs)
    assert cvl_x0.io_backend.kwargs["a"] == "b"
    assert cvl_x0.io_backend.kwargs["cloudpath"] == kwargs["path"]
    assert cvl_x0.readonly == kwargs["readonly"]
    assert cvl_x0.index_converter.index_resolution == kwargs["index_resolution"]
    assert cvl_x0.index_converter.allow_rounding == kwargs["allow_shape_rounding"]
    assert (
        cvl_x0.index_converter.default_desired_resolution == kwargs["default_desired_resolution"]
    )
    assert cvl_x0.index_adjs == kwargs["index_adjs"]
    assert cvl_x0.read_postprocs == kwargs["read_postprocs"]
    assert cvl_x0.write_preprocs == kwargs["write_preprocs"]

    kwargs["interpolation_mode"] = "img"
    kwargs["data_resolution"] = (3, 3, 3)

    cvl_x1 = build_cv_layer(**kwargs)
    assert len(cvl_x1.index_adjs) == 1 + len(kwargs["index_adjs"])
    assert cvl_x1.index_adjs[0].data_resolution == kwargs["data_resolution"]
    assert cvl_x1.index_adjs[0].interpolation_mode == kwargs["interpolation_mode"]
    assert cvl_x1.index_adjs[0].allow_rounding == kwargs["allow_shape_rounding"]


def test_layer_set(mocker):
    kwargs = {
        "layers": {
            "layer1": mocker.Mock(),
            "layer2": mocker.Mock(),
        }
    }
    layer_set = build_layer_set(**kwargs)
    assert layer_set.io_backend.layers == kwargs["layers"]

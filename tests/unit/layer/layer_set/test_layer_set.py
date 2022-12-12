# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
from zetta_utils.layer import build_layer_set
from zetta_utils.layer.layer_set import LayerSetIndex

# TODO: refactor these tests to use less mocks, use parametrize


def test_constructor(mocker):
    layer1_m = mocker.MagicMock()
    layer2_m = mocker.MagicMock()
    build_layer_set(
        layers={
            "1": layer1_m,
            "2": layer2_m,
        }
    )


def test_read_all(mocker):
    layers = {}
    for key in ["1", "2"]:
        layer_m = mocker.MagicMock()
        data_m = mocker.MagicMock()
        layer_m.read = mocker.MagicMock(return_value=data_m)
        layers[key] = layer_m

    lset = build_layer_set(layers=layers)
    idx_m = mocker.MagicMock()
    result = lset.read(idx_m)
    for k, v in layers.items():
        assert k in result
        assert result[k] == v.read.return_value
        v.read.assert_called_with(idx_m)


def test_read_select_naive(mocker):
    layers = {}
    for key in ["1", "2"]:
        layer_m = mocker.MagicMock()
        data_m = mocker.MagicMock()
        layer_m.read = mocker.MagicMock(return_value=data_m)
        layers[key] = layer_m

    lset = build_layer_set(layers=layers)
    idx_m = mocker.MagicMock()
    result = lset.read((("1",), idx_m))

    assert "1" in result
    assert result["1"] == layers["1"].read.return_value
    layers["1"].read.assert_called_with(idx_m)
    assert "2" not in result


def test_read_select_complex(mocker) -> None:
    layers = {}
    for key in ["1", "2"]:
        layer_m = mocker.MagicMock()
        data_m = mocker.MagicMock()
        layer_m.read = mocker.MagicMock(return_value=data_m)
        layers[key] = layer_m

    lset = build_layer_set(layers=layers)
    idx_m = mocker.MagicMock()
    result = lset.read((("1",), idx_m, idx_m))

    assert "1" in result
    assert result["1"] == layers["1"].read.return_value
    layers["1"].read.assert_called_with((idx_m, idx_m))
    assert "2" not in result


def test_read_select_complex_backend_index(mocker) -> None:
    layers = {}
    for key in ["1", "2"]:
        layer_m = mocker.MagicMock()
        data_m = mocker.MagicMock()
        layer_m.read = mocker.MagicMock(return_value=data_m)
        layers[key] = layer_m

    lset = build_layer_set(layers=layers)
    idx_m = mocker.MagicMock()
    result = lset.read(LayerSetIndex(layer_selection=("1",), layer_idx=(idx_m, idx_m)))

    assert "1" in result
    assert result["1"] == layers["1"].read.return_value
    layers["1"].read.assert_called_with((idx_m, idx_m))
    assert "2" not in result


def test_write_all(mocker):
    layers = {}
    datas = {}
    for key in ["1", "2"]:
        layer_m = mocker.MagicMock()
        data_m = mocker.MagicMock()
        layer_m.write = mocker.MagicMock()
        datas[key] = data_m
        layers[key] = layer_m

    lset = build_layer_set(layers=layers)
    idx_m = mocker.MagicMock()
    lset.write(idx_m, datas)

    for k, v in layers.items():
        v.write.assert_called_with(idx_m, datas[k])

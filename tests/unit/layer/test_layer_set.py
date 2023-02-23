from zetta_utils.layer.layer_set import build_layer_set


def test_read(mocker):
    layer_a = mocker.MagicMock()
    layer_a.read_with_procs = mocker.MagicMock(return_value="a")
    layer_b = mocker.MagicMock()
    layer_b.read_with_procs = mocker.MagicMock(return_value="b")
    layer_set = build_layer_set(layers={"a": layer_a, "b": layer_b})
    idx = mocker.MagicMock()
    result = layer_set[idx]
    assert result == {"a": "a", "b": "b"}
    layer_a.read_with_procs.called_with(idx=idx)
    layer_b.read_with_procs.called_with(idx=idx)


def test_write(mocker):
    layer_a = mocker.MagicMock()
    layer_a.write_with_procs = mocker.MagicMock()
    layer_b = mocker.MagicMock()
    layer_b.write_with_procs = mocker.MagicMock()
    layer_set = build_layer_set(layers={"a": layer_a, "b": layer_b})
    idx = mocker.MagicMock()
    layer_set[idx] = {"a": 1, "b": 2}
    layer_a.write_with_procs.called_with(idx=idx, data=1)
    layer_b.write_with_procs.called_with(idx=idx, data=2)

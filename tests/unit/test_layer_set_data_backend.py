# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
from zetta_utils.io.backends import LayerSetBackend
from zetta_utils.io.indexes import SetSelectionIndex


def test_backend_get_index_type():
    index_type = LayerSetBackend.get_index_type()
    assert index_type == SetSelectionIndex


def test_set_backend_read(mocker):
    layer0 = mocker.Mock()
    layer1 = mocker.Mock()
    layer0.read = mocker.Mock()
    layer1.read = mocker.Mock()
    lsb = LayerSetBackend(layers={"1": layer1, "0": layer0})

    idx0 = (1, 2, 3)
    set_idx0 = SetSelectionIndex.convert((None, idx0))
    lsb.read(set_idx0)
    layer0.read.assert_called_with(idx0)
    layer1.read.assert_called_with(idx0)

    idx1 = (3, 4, 5)
    set_idx0 = SetSelectionIndex.convert(("1", idx1))
    lsb.read(set_idx0)
    layer0.read.assert_called_with(idx0)
    layer1.read.assert_called_with(idx1)


def test_set_backend_write(mocker):
    layer0 = mocker.Mock()
    layer1 = mocker.Mock()
    layer0.write = mocker.Mock()
    layer1.write = mocker.Mock()
    lsb = LayerSetBackend(layers={"1": layer1, "0": layer0})

    idx0 = (1, 2, 3)
    value0 = {"1": 1, "0": 0}
    set_idx0 = SetSelectionIndex.convert((None, idx0))
    lsb.write(set_idx0, value0)
    layer0.write.assert_called_with(idx0, value0["0"])
    layer1.write.assert_called_with(idx0, value0["1"])

# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access
import pytest

from zetta_utils.io.layer import Layer
from zetta_utils.io.indexes import IndexAdjusterWithProcessors


def test_convert_index(mocker):
    dummy_backend = mocker.MagicMock()
    dummy_index_type = mocker.MagicMock()
    dummy_index = mocker.MagicMock()
    dummy_index_type.convert = mocker.MagicMock(return_value=dummy_index)
    dummy_backend.get_index_type = mocker.MagicMock(return_value=dummy_index_type)
    layer = Layer(dummy_backend)
    idx_raw = mocker.MagicMock()
    result = layer._convert_index(idx_raw)
    assert result == dummy_index
    dummy_index_type.convert.assert_called_with(idx_raw)

    dummy_indexer_index = mocker.MagicMock()
    dummy_indexer = mocker.MagicMock(return_value=dummy_indexer_index)
    layer.index_converter = dummy_indexer

    result_indexer = layer._convert_index(idx_raw)
    assert result_indexer == dummy_indexer_index
    dummy_indexer.assert_called_with(idx_raw)


def test_apply_index_adjs(mocker):
    dummy_backend = mocker.MagicMock()
    layer = Layer(dummy_backend)

    idx0 = mocker.MagicMock()
    idx1 = mocker.MagicMock()
    idx2 = mocker.MagicMock()

    processor = mocker.MagicMock()

    idx_adj0 = mocker.MagicMock(return_value=idx1)
    idx_adj1 = mocker.MagicMock(spec=IndexAdjusterWithProcessors, return_value=(idx2, [processor]))
    layer.index_adjs = [idx_adj0, idx_adj1]

    result1 = layer._apply_index_adjs(idx0, "read")
    assert result1 == (idx2, [processor])
    idx_adj0.assert_called_with(idx0)
    idx_adj1.assert_called_with(idx1, mode="read")


def test_read(mocker):
    idx0 = mocker.MagicMock()
    idx1 = mocker.MagicMock()
    idx2 = mocker.MagicMock()

    value0 = mocker.MagicMock()
    value1 = mocker.MagicMock()
    value2 = mocker.MagicMock()

    dummy_backend = mocker.MagicMock()
    dummy_backend.read = mocker.MagicMock(return_value=value0)

    proc0 = mocker.MagicMock(return_value=value1)
    proc1 = mocker.MagicMock(return_value=value2)

    layer = Layer(dummy_backend, read_postprocs=[proc1])

    mocker.patch("zetta_utils.io.layer.Layer._convert_index", return_value=idx1)
    mocker.patch("zetta_utils.io.layer.Layer._apply_index_adjs", return_value=(idx2, [proc0]))

    result = layer.read(idx0)
    assert result == value2
    dummy_backend.read.assert_called_with(idx=idx2)
    proc1.assert_called_with(data=value1)


def test_write(mocker):
    idx0 = mocker.MagicMock()
    idx1 = mocker.MagicMock()
    idx2 = mocker.MagicMock()

    value0 = mocker.MagicMock()
    value1 = mocker.MagicMock()
    value2 = mocker.MagicMock()

    dummy_backend = mocker.MagicMock()
    dummy_backend.write = mocker.MagicMock()

    proc0 = mocker.MagicMock(return_value=value1)
    proc1 = mocker.MagicMock(return_value=value2)

    layer = Layer(dummy_backend, write_preprocs=[proc1])

    mocker.patch("zetta_utils.io.layer.Layer._convert_index", return_value=idx1)
    mocker.patch("zetta_utils.io.layer.Layer._apply_index_adjs", return_value=(idx2, [proc0]))

    layer.write(idx0, value0)

    dummy_backend.write.assert_called_with(idx=idx2, value=value2)
    proc1.assert_called_with(data=value1)


def test_write_exc(mocker):
    dummy_backend = mocker.Mock()
    layer = Layer(dummy_backend, readonly=True)
    with pytest.raises(IOError):
        layer.write(None, None)

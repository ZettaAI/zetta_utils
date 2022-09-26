# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access
import pytest

from zetta_utils.io.layer import Layer
from zetta_utils.io.indexes import IndexAdjusterWithProcessors


def test_read_no_conv(mocker):
    idx_in_m = mocker.MagicMock()

    data_read_raw_m = mocker.MagicMock()
    dummy_backend = mocker.MagicMock()
    dummy_backend.read = mocker.MagicMock(return_value=data_read_raw_m)
    index_type_m = mocker.MagicMock()
    idx_conv_m = mocker.MagicMock()
    index_type_m.default_convert = mocker.MagicMock(return_value=idx_conv_m)
    dummy_backend.get_index_type = mocker.MagicMock(return_value=index_type_m)

    layer = Layer(
        dummy_backend,
    )

    result = layer.read(idx_in_m)
    dummy_backend.read.assert_called_with(idx=idx_conv_m)
    assert result == data_read_raw_m


def test_read(mocker):
    idx_in_m = mocker.MagicMock()

    idx_conved_m = mocker.MagicMock()
    index_converter_m = mocker.MagicMock(return_value=idx_conved_m)

    idx_adjed_m = mocker.MagicMock()
    index_adj_m = mocker.MagicMock(return_value=idx_adjed_m)

    data_read_raw_m = mocker.MagicMock()
    dummy_backend = mocker.MagicMock()
    dummy_backend.read = mocker.MagicMock(return_value=data_read_raw_m)

    data_proced_m = mocker.MagicMock()
    read_proc_m = mocker.MagicMock(return_value=data_proced_m)

    layer = Layer(
        dummy_backend,
        index_converter=index_converter_m,
        read_postprocs=[read_proc_m],
        index_adjs=[index_adj_m],
    )

    result = layer.read(idx_in_m)
    index_converter_m.assert_called_with(idx_raw=idx_in_m)
    index_adj_m.assert_called_with(idx=idx_conved_m)
    dummy_backend.read.assert_called_with(idx=idx_adjed_m)
    read_proc_m.assert_called_with(data=data_read_raw_m)
    assert result == data_proced_m


def test_read_adj_w_proc(mocker):
    idx_in_m = mocker.MagicMock()

    idx_adjed_m = mocker.MagicMock()
    data_idx_proced_m = mocker.MagicMock()
    idx_data_proc_m = mocker.MagicMock(return_value=data_idx_proced_m)
    index_adj_m = mocker.MagicMock(
        return_value=(idx_adjed_m, [idx_data_proc_m]), spec=IndexAdjusterWithProcessors
    )

    data_read_raw_m = mocker.MagicMock()
    dummy_backend = mocker.MagicMock()
    dummy_backend.read = mocker.MagicMock(return_value=data_read_raw_m)

    data_proced_m = mocker.MagicMock()
    read_proc_m = mocker.MagicMock(return_value=data_proced_m)

    layer = Layer(
        dummy_backend,
        read_postprocs=[read_proc_m],
        index_adjs=[index_adj_m],
    )

    result = layer.read(idx_in_m)
    dummy_backend.read.assert_called_with(idx=idx_adjed_m)
    idx_data_proc_m.assert_called_with(data=data_read_raw_m)
    read_proc_m.assert_called_with(data=data_idx_proced_m)
    assert result == data_proced_m


def test_write_exc(mocker):
    dummy_backend = mocker.Mock()
    layer = Layer(dummy_backend, readonly=True)
    with pytest.raises(IOError):
        layer.write(None, None)


def test_write_adj_w_proc(mocker):
    idx_in_m = mocker.MagicMock()
    data_in_m = mocker.MagicMock()

    idx_adjed_m = mocker.MagicMock()
    data_idx_proced_m = mocker.MagicMock()
    idx_data_proc_m = mocker.MagicMock(return_value=data_idx_proced_m)
    index_adj_m = mocker.MagicMock(
        return_value=(idx_adjed_m, [idx_data_proc_m]), spec=IndexAdjusterWithProcessors
    )

    dummy_backend = mocker.MagicMock()
    dummy_backend.write = mocker.MagicMock()

    data_proced_m = mocker.MagicMock()
    write_proc_m = mocker.MagicMock(return_value=data_proced_m)

    layer = Layer(
        dummy_backend,
        write_preprocs=[write_proc_m],
        index_adjs=[index_adj_m],
    )

    layer.write(idx_in_m, data_in_m)
    idx_data_proc_m.assert_called_with(data=data_in_m)
    write_proc_m.assert_called_with(data=data_idx_proced_m)

    dummy_backend.write.assert_called_with(idx=idx_adjed_m, value=data_proced_m)

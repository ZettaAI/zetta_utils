# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long
import pytest

# from unittest import mock

from zetta_utils.data.layers.common import Layer


@pytest.fixture
def patch_layer_repr(mocker):
    mocker.patch("zetta_utils.data.layers.common.Layer.__repr__", return_value="Layer")


def test_write_exc(mocker, patch_layer_repr):
    layer = Layer(readonly=True)
    with pytest.raises(IOError):
        layer.write(None, None)


def test_write(mocker):
    mocked_inner_write = mocker.patch("zetta_utils.data.layers.common.Layer._write")
    layer = Layer(readonly=False)
    layer.write(123, "abc")
    mocked_inner_write.assert_called_once()


def dummy():
    pass


@pytest.mark.parametrize(
    "idx, adjs, pprocs, expected_idx, expected_out",
    [
        [[], None, None, [], "a"],
        [[], [lambda x: x + [1]], None, [1], "a"],
        [[], [lambda x: x + [1], lambda x: x + [2]], None, [1, 2], "a"],
        [[], None, [lambda x: x + "b"], [], "ab"],
        [[], None, [lambda x: x + "b", lambda x: x + "c"], [], "abc"],
    ],
)
def test_read(idx, adjs, pprocs, expected_idx, expected_out, mocker):
    mocked_inner_read = mocker.patch(
        "zetta_utils.data.layers.common.Layer._read", return_value="a"
    )
    layer = Layer(
        read_index_adjs=adjs,
        read_postprocs=pprocs,
    )
    result = layer.read(idx)
    assert result == expected_out
    mocked_inner_read.assert_called_once()
    mocked_inner_read.assert_called_with(idx=expected_idx)

# pylint: disable=missing-docstring

import pytest

from zetta_utils.layer.db_layer import DBFrontend, build_db_layer


def test_write_scalar(mocker) -> None:
    backend = mocker.MagicMock()
    backend.write = mocker.MagicMock()

    layer = build_db_layer(backend)

    layer["key"] = "val"
    assert backend.write.call_args.kwargs["data"] == [{"value": "val"}]

    layer["key"] = ["val"]  # this should not work but does
    assert backend.write.call_args.kwargs["data"] == [{"value": "val"}]


def test_write_list(mocker) -> None:
    backend = mocker.MagicMock()
    backend.write = mocker.MagicMock()

    layer = build_db_layer(backend)

    idx_user = ["key0", "key1"]
    data_user = ["val0", "val1"]
    layer[idx_user] = data_user
    assert backend.write.call_args.kwargs["data"] == [{"value": "val0"}, {"value": "val1"}]


def test_write_single_row(mocker) -> None:
    backend = mocker.MagicMock()
    backend.write = mocker.MagicMock()

    layer = build_db_layer(backend)

    row_key = "key"
    col_keys = ("col0", "col1")
    idx_user = (row_key, col_keys)

    data_user = {
        "col0": "val0",
        "col1": "val1",
    }

    layer[idx_user] = data_user
    assert backend.write.call_args.kwargs["data"] == [{"col0": "val0", "col1": "val1"}]


def test_write_rows(mocker) -> None:
    backend = mocker.MagicMock()
    backend.write = mocker.MagicMock()
    layer = build_db_layer(backend)

    row_keys = ["key0", "key1"]
    idx_user = (row_keys, ("col0", "col1"))

    data_user = [
        {"col0": "val0", "col1": "val1"},
        {"col0": "val0"},
    ]

    layer[idx_user] = data_user
    assert backend.write.call_args.kwargs["data"] == data_user


def test_write_exc(mocker):
    backend = mocker.MagicMock()
    backend.write = mocker.MagicMock()

    layer = build_db_layer(backend)
    with pytest.raises(TypeError):
        layer["key"] = object

    with pytest.raises(ValueError):
        layer["key"] = mocker.MagicMock()


@pytest.mark.parametrize(
    "idx_user, data, expected",
    [
        [
            "key0",
            [{"value": "val0"}],
            "val0",
        ],
        [
            "key42",
            [{"value": 42}],
            42,
        ],
        [
            ["key0", "key42"],
            [{"value": "val0"}, {"value": 42}],
            ["val0", 42],
        ],
        [
            ("key1", ("col0", "col1")),
            [{"col0": "val0", "col1": "val1"}],
            {"col0": "val0", "col1": "val1"},
        ],
        [
            (["key1", "key2"], ("col0", "col1")),
            [{"col0": "val0", "col1": "val1"}, {"col0": None, "col1": "val2"}],
            [{"col0": "val0", "col1": "val1"}, {"col0": None, "col1": "val2"}],
        ],
    ],
)
def test_db_read_convert(
    idx_user,
    data,
    expected,
):
    frontend = DBFrontend()
    result = frontend.convert_read_data(idx_user, data)
    assert result == expected

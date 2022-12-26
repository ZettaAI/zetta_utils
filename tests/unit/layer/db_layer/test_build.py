# pylint: disable=missing-docstring

from zetta_utils.layer.db_layer import build_db_layer


def test_write_scalar(mocker) -> None:
    backend = mocker.MagicMock()
    backend.write = mocker.MagicMock()

    layer = build_db_layer(backend)

    layer["key"] = "val"
    assert backend.write.call_args.kwargs["data"] == [{"value": "val"}]


def test_write_list(mocker) -> None:
    backend = mocker.MagicMock()
    backend.write = mocker.MagicMock()

    layer = build_db_layer(backend)

    user_idx = ["key0", "key1"]
    user_dat = ["val0", "val1"]
    layer[user_idx] = user_dat
    assert backend.write.call_args.kwargs["data"] == [{"value": "val0"}, {"value": "val1"}]

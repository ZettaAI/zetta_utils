# pylint: disable=missing-docstring

from zetta_utils.layer.db_layer import build_db_layer


def test_write_scalar(mocker):
    backend = mocker.MagicMock()
    backend.write = mocker.MagicMock()

    layer = build_db_layer(backend)

    layer["key"] = "val"
    assert backend.write.call_args.kwargs["data"] == [{"value": "val"}]

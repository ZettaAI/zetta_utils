# pylint: disable=missing-docstring,protected-access
import pytest

from zetta_utils.layer.db_layer import DBIndex, DBLayer


@pytest.mark.parametrize(
    "idx, expected",
    [
        [
            DBIndex({}),
            DBIndex({}),
        ],
        [
            "key",
            DBIndex({"key": ("value",)}),
        ],
        [
            ("key0", ("col0", "col1")),
            DBIndex({"key0": ("col0", "col1")}),
        ],
        [
            (["key0", "key1"], ("col0", "col1")),
            DBIndex({"key0": ("col0", "col1"), "key1": ("col0", "col1")}),
        ],
    ],
)
def test_db_convert(
    idx,
    expected: DBIndex,
    mocker,
):
    layer = DBLayer(backend=mocker.MagicMock())
    result = layer._convert_idx(idx)
    assert result.get_size() == expected.get_size()
    assert result.row_keys == expected.row_keys
    assert result.rows_col_keys == expected.rows_col_keys

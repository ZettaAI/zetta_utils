# pylint: disable=missing-docstring,protected-access
import pytest

from zetta_utils.layer.db_layer import DBFrontend, DBIndex


@pytest.mark.parametrize(
    "kwargs, idx, expected",
    [
        [
            {},
            DBIndex({}),
            DBIndex({}),
        ],
        [
            {},
            "key",
            DBIndex({"key": ("value",)}),
        ],
        [
            {},
            ("key0", ("col0", "col1")),
            DBIndex({"key0": ("col0", "col1")}),
        ],
        [
            {},
            (["key0", "key1"], ("col0", "col1")),
            DBIndex({"key0": ("col0", "col1"), "key1": ("col0", "col1")}),
        ],
    ],
)
def test_volumetric_convert(
    kwargs: dict,
    idx,
    expected: DBIndex,
):
    convert = DBFrontend(**kwargs)
    result = convert._convert_idx(idx)
    assert result.get_size() == expected.get_size()
    assert result.row_keys == expected.row_keys
    assert result.col_keys == expected.col_keys

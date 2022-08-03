# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
import pytest

from zetta_utils.io.indexes.set_selection import SetSelectionIndex, RawSetSelectionIndex


@pytest.mark.parametrize(
    "idx_raw, expected",
    [
        [
            (42,),
            SetSelectionIndex(layer_selection=None, layer_idx=(42,)),
        ],
        [
            (None, 42),
            SetSelectionIndex(layer_selection=None, layer_idx=(None, 42)),
        ],
        [
            ("123", 42),
            SetSelectionIndex(layer_selection=None, layer_idx=("123", 42)),
        ],
        [
            (("123", "456"), 42),
            SetSelectionIndex(layer_selection=("123", "456"), layer_idx=(42,)),
        ],
    ],
)
def test_set_selection_index(idx_raw: RawSetSelectionIndex, expected: SetSelectionIndex):
    result = SetSelectionIndex.convert(idx_raw)
    assert result == expected

# pylint: disable=redefined-outer-name
import pytest

from zetta_utils.augmentations import prob_aug


@pytest.fixture
def dummy_aug():
    return prob_aug(lambda x: x)


def test_args_with_data_kwarg_exc(dummy_aug):
    with pytest.raises(Exception):
        dummy_aug("input", "other", data="conflicting_data")


def test_no_data_exc(dummy_aug):
    with pytest.raises(Exception):
        dummy_aug(x="x")

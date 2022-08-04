# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
import pytest
import numpy as np

from zetta_utils.io.backends import CVBackend
from zetta_utils.io.indexes import VolumetricIndex


def test_backend_get_index_type():
    index_type = CVBackend.get_index_type()
    assert index_type == VolumetricIndex


def test_cv_backend_constructor():
    cvb = CVBackend(bounded=True)
    assert cvb.kwargs["bounded"]
    assert not cvb.kwargs["autocrop"]


def test_cv_backend_constructor_exc():
    with pytest.raises(ValueError):
        CVBackend(mip=4)


def test_get_cv_at_res(mocker):
    expected = mocker.Mock()
    cv_fn = mocker.patch(
        "zetta_utils.io.backends.cv.CachedCloudVolume",
        return_value=expected,
    )
    cvb = CVBackend()
    result = cvb._get_cv_at_resolution((1, 1, 1))
    assert result == expected
    cv_fn.assert_called_with(mip=(1, 1, 1), **cvb.kwargs)


# Don't know how to make a better dummy
class DummyCV:
    def __init__(self, expected):
        self.expected = expected
        self.last_call_args = None
        self.last_call_kwargs = None

    def __getitem__(self, *args, **kwargs):
        self.last_call_args = args
        self.last_call_kwargs = kwargs
        return self.expected

    def __setitem__(self, *args, **kwargs):
        self.last_call_args = args
        self.last_call_kwargs = kwargs
        return self.expected


def test_cv_backend_read(mocker):
    data_read = np.ones([3, 4, 5, 2])
    expected = np.ones([2, 3, 4, 5])

    dummy = DummyCV(data_read)
    mocker.patch("zetta_utils.io.backends.cv.CVBackend._get_cv_at_resolution", return_value=dummy)
    cvb = CVBackend()
    index = VolumetricIndex(slices=(slice(0, 1), slice(1, 2), slice(2, 3)), resolution=(5, 6, 7))
    result = cvb.read(index)
    np.testing.assert_array_equal(result, expected)
    assert dummy.last_call_args == (index.slices,)


def test_cv_backend_write(mocker):
    dummy = DummyCV(None)
    mocker.patch("zetta_utils.io.backends.cv.CVBackend._get_cv_at_resolution", return_value=dummy)
    cvb = CVBackend()
    value = np.ones([2, 3, 4, 5])
    expected_written = np.ones([3, 4, 5, 2])
    index = VolumetricIndex(slices=(slice(0, 1), slice(1, 2), slice(2, 3)), resolution=(5, 6, 7))
    cvb.write(index, value)
    assert dummy.last_call_args[0] == index.slices
    np.testing.assert_array_equal(dummy.last_call_args[1], expected_written)


@pytest.mark.parametrize(
    "data_in,expected_exc",
    [
        [np.ones((1, 2, 3, 4, 5, 6)), ValueError],
    ],
)
def test_cv_backend_write_exc(data_in, expected_exc, mocker):
    dummy = DummyCV(None)
    mocker.patch("zetta_utils.io.backends.cv.CVBackend._get_cv_at_resolution", return_value=dummy)
    cvb = CVBackend()
    index = VolumetricIndex(slices=(slice(0, 1), slice(1, 2), slice(2, 3)), resolution=(5, 6, 7))
    with pytest.raises(expected_exc):
        cvb.write(index, data_in)

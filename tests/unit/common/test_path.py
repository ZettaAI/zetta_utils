import os

import pytest

from zetta_utils.common.path import abspath


@pytest.mark.parametrize(
    "path, expected",
    [
        ["abc", "file://" + os.getcwd() + "/abc"],
        ["./abc", "file://" + os.getcwd() + "/abc"],
        ["gs://abc", "gs://abc"],
        ["~/abc", "file://" + os.path.expanduser("~") + "/abc"],
        ["../abc", "file://" + os.path.dirname(os.getcwd()) + "/abc"],
        ["file://../abc", "file://" + os.path.dirname(os.getcwd()) + "/abc"],
    ],
)
def test_path(path, expected):
    assert abspath(path) == expected

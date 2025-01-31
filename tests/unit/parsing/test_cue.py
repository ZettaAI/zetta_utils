# pylint: disable=missing-docstring,unspecified-encoding,invalid-name
import pathlib

import pytest

from zetta_utils.parsing import cue

THIS_DIR = pathlib.Path(__file__).parent.resolve()
TEST_DATA_DIR = THIS_DIR / "../assets/params/"
TEST_FILE_PATH = TEST_DATA_DIR / "file_x0.cue"
UNEXISTING_FILE_PATH = TEST_DATA_DIR / "no_file_x0.cue"


def test_loads():
    result = cue.loads('["1", "1"]')
    assert result == ["1", "1"]


def test_load_posix_path():
    result = cue.load(TEST_FILE_PATH)
    assert result == {"one": "two", "three": "four"}


def test_load_str_path():
    result = cue.load(str(TEST_FILE_PATH))
    assert result == {"one": "two", "three": "four"}


def test_load_fp():
    with open(TEST_FILE_PATH, "r") as fp:
        result = cue.load(fp)
        assert result == {"one": "two", "three": "four"}


@pytest.mark.parametrize("inp", [[str(UNEXISTING_FILE_PATH)], [333]])
def test_load_nonexist(inp):
    with pytest.raises(Exception):
        cue.load(inp)

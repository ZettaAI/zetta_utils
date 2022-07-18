# pylint: disable=missing-docstring,unspecified-encoding,invalid-name
import pathlib
import pytest

import zetta_utils as zu

THIS_DIR = pathlib.Path(__file__).parent.resolve()
TEST_DATA_DIR = THIS_DIR / "data/params/"
TEST_FILE_PATH = TEST_DATA_DIR / "file_x0.cue"
UNEXISTING_FILE_PATH = TEST_DATA_DIR / "no_file_x0.cue"


def test_load_posix_path():
    result = zu.cue.load(TEST_FILE_PATH)
    assert result == {"one": "two", "three": "four"}


def test_load_str_path():
    result = zu.cue.load(str(TEST_FILE_PATH))
    assert result == {"one": "two", "three": "four"}


def test_load_fp():
    with open(TEST_FILE_PATH, "r") as fp:
        result = zu.cue.load(fp)
        assert result == {"one": "two", "three": "four"}


@pytest.mark.parametrize("inp", [[str(UNEXISTING_FILE_PATH)], [333]])
def test_load_nonexist(inp):
    with pytest.raises(Exception):
        zu.cue.load(inp)

# pylint: disable=unused-argument,redefined-outer-name
import tempfile

import pytest
from click.testing import CliRunner

from zetta_utils import builder, cli
from zetta_utils.parsing import json


def dummy(i):
    return i


@pytest.fixture
def register_dummy():
    builder.register("dummy")(dummy)
    yield
    del builder.REGISTRY["dummy"]


@pytest.mark.parametrize(
    "spec",
    [
        {"@type": "dummy", "i": {"a": "b"}},
    ],
)
def test_zetta_run(spec, register_dummy, mocker, firestore_emulator):
    mocker.patch("zetta_utils.parsing.cue.load", return_value=spec)
    runner = CliRunner()

    mocker.patch("fsspec.open", return_value=tempfile.TemporaryFile(mode="w"))
    result = runner.invoke(cli.run, ".")
    assert result.exit_code == 0

    mocker.patch("fsspec.open", return_value=tempfile.TemporaryFile(mode="w"))
    result = runner.invoke(cli.run, ["-p", "."])
    assert result.exit_code == 0


def test_show_registry(register_dummy):
    runner = CliRunner()
    result = runner.invoke(cli.show_registry)
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "spec",
    [
        {"@type": "dummy", "i": {"a": "b"}},
    ],
)
def test_zetta_run_str(spec, register_dummy, mocker, firestore_emulator):
    mocker.patch(
        "fsspec.open",
        side_effect=[tempfile.TemporaryFile(mode="w"), tempfile.TemporaryFile(mode="w")],
    )
    runner = CliRunner()
    result = runner.invoke(cli.run, ["-s", json.dumps(spec)])
    assert result.exit_code == 0


CUSTOM_IMPORT_CONTENT = """
from zetta_utils import builder
builder.register("registered_in_file")(lambda: None)
"""
SPEC_WITH_CUSTOM_IMPORT = {"@type": "registered_in_file"}


def test_zetta_run_extra_import_fail(tmp_path, firestore_emulator):
    runner = CliRunner()
    # make sure that it doesn't run without a file import
    should_fail = runner.invoke(cli.run, ["-s", json.dumps(SPEC_WITH_CUSTOM_IMPORT)])
    assert should_fail.exit_code != 0


def test_zetta_run_extra_import_success(tmp_path, mocker, firestore_emulator):
    mocker.patch(
        "fsspec.open",
        side_effect=[tempfile.TemporaryFile(mode="w"), tempfile.TemporaryFile(mode="w")],
    )
    runner = CliRunner()
    my_file = tmp_path / "custom_import.py"
    my_file.write_text(CUSTOM_IMPORT_CONTENT)
    should_succeed = runner.invoke(
        cli.run, ["-s", json.dumps(SPEC_WITH_CUSTOM_IMPORT), "-i", str(my_file)]
    )
    assert should_succeed.exit_code == 0


def test_zetta_run_extra_import_py_check_fail(tmp_path, firestore_emulator):
    runner = CliRunner()
    my_file = tmp_path / "custom_import"
    my_file.write_text(CUSTOM_IMPORT_CONTENT)
    should_succeed = runner.invoke(
        cli.run, ["-s", json.dumps(SPEC_WITH_CUSTOM_IMPORT), "-i", str(my_file)]
    )
    assert should_succeed.exit_code != 0

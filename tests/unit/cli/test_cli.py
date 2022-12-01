# pylint: disable=unused-argument,redefined-outer-name
import json

import pytest
from click.testing import CliRunner

from zetta_utils import builder, cli


@pytest.fixture
def register_dummy():
    def dummy(i):
        return i

    builder.register("dummy")(dummy)
    yield
    del builder.REGISTRY["dummy"]


@pytest.mark.parametrize(
    "spec",
    [
        {"@type": "dummy", "i": {"a": "b"}},
    ],
)
def test_zetta_run(spec, register_dummy, mocker):
    mocker.patch("zetta_utils.parsing.cue.load", return_value=spec)
    runner = CliRunner()
    result = runner.invoke(cli.run, ".")
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
def test_zetta_run_str(spec, register_dummy):
    runner = CliRunner()
    result = runner.invoke(cli.run, ["-s", json.dumps(spec)])
    assert result.exit_code == 0

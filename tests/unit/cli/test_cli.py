# pylint: disable=unused-argument,redefined-outer-name
import json
import pytest
from click.testing import CliRunner

import zetta_utils as zu
from zetta_utils import cli  # pylint: disable=unused-import


@pytest.fixture
def register_dummy():
    def dummy(i):
        return i

    zu.builder.register("dummy")(dummy)
    yield
    del zu.builder.REGISTRY["dummy"]


@pytest.mark.parametrize(
    "spec, expected_output",
    [
        [
            {"@type": "dummy", "i": {"a": "b"}},
            {"a": "b"},
        ],
    ],
)
def test_zetta_run(spec, expected_output, register_dummy, mocker):
    mocker.patch("zetta_utils.parsing.cue.load", return_value=spec)
    runner = CliRunner()
    result = runner.invoke(zu.cli.run, ".")
    assert result.exit_code == 0
    assert json.loads(result.output.replace("'", '"')) == expected_output


def test_show_registry(register_dummy):
    runner = CliRunner()
    result = runner.invoke(zu.cli.show_registry)
    assert result.exit_code == 0
    assert "dummy" in result.output

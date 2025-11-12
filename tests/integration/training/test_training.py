# pylint: disable=line-too-long, unused-import
import os
import pathlib
import sys
import tempfile
import time

import pytest

import zetta_utils
from zetta_utils import run

zetta_utils.load_all_modules()


@pytest.mark.skipif(
    "not config.getoption('--run-integration')",
    reason="Only run when `--run-integration` is given",
)
def test_ci_training():
    """Test that the CI training configuration can be loaded and executed."""
    # Get Python version (e.g., "3.12" -> "py312")
    python_version = f"py{sys.version_info.major}{sys.version_info.minor}"

    # Read the template CUE file (relative to this test file)
    test_dir = pathlib.Path(__file__).parent.resolve()
    cue_template_path = test_dir / "ci_training.cue"
    with open(cue_template_path, "r", encoding="utf-8") as f:
        cue_content = f.read()

    # Replace the image tag with the correct Python version
    cue_content = cue_content.replace("PYTHON_VERSION", python_version)

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cue", delete=False, encoding="utf-8"
    ) as f:
        f.write(cue_content)
        temp_cue_path = f.name

    try:
        spec = zetta_utils.parsing.cue.load(temp_cue_path)
        # Set RUN_ID with timestamp to make it unique for each test run
        run.RUN_ID = f"ci-test-{python_version}-{int(time.time())}"
        try:
            result = zetta_utils.builder.build(spec)
            assert result is not None
        finally:
            run.RUN_ID = None
        del spec
    finally:
        # Clean up temp file
        os.unlink(temp_cue_path)

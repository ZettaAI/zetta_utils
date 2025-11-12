# pylint: disable=line-too-long, unused-import
import os
import pathlib
import sys
import tempfile

import pytest

import zetta_utils
from zetta_utils import constants, run

zetta_utils.load_all_modules()


@pytest.mark.skipif(
    "not config.getoption('--run-integration')",
    reason="Only run when `--run-integration` is given",
)
def test_ci_training():
    """Test that the CI training configuration can be loaded and executed."""
    # Disable run database for CI testing (project uses Datastore mode, not native Firestore)
    original_run_db = constants.RUN_DATABASE
    constants.RUN_DATABASE = None

    try:
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
            # Initialize run context for training (without database tracking)
            with run.run_ctx_manager(main_run_process=True, spec=spec):
                result = zetta_utils.builder.build(spec)
                assert result is not None
            del spec
        finally:
            # Clean up temp file
            os.unlink(temp_cue_path)
    finally:
        # Restore original RUN_DATABASE setting
        constants.RUN_DATABASE = original_run_db

"""cuelang parsing."""
import json
import os
import pathlib
import subprocess
import tempfile

import fsspec

from zetta_utils import log

logger = log.get_logger("zetta_utils")

cue_exe = os.environ.get("CUE_EXE", "cue")


def loads(s: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_tmp_path = os.path.join(tmp_dir, "tmp_spec.cue")
        with open(local_tmp_path, "w", encoding="utf8") as tmp_f:
            tmp_f.write(s)
        result = load_local(local_tmp_path)
    return result


def load_local(local_path: str):
    local_path_str = _to_str_path(local_path)
    command_result = subprocess.run(
        [cue_exe, "export", local_path_str], capture_output=True, check=False
    )

    if command_result.returncode != 0:
        raise RuntimeError(  # pragma: no cover
            f"CUE failed parsing {local_path_str}: {str(command_result.stderr)}"
        )

    result_str = command_result.stdout
    result = json.loads(result_str)
    return result


def _to_str_path(path):
    if isinstance(path, (str, pathlib.PosixPath)):
        result = str(path)
    elif hasattr(path, "name"):  # File pointers
        result = path.name
    else:
        raise ValueError(f"Invalid input path: {path}")
    return result


def load(path):
    path_str = _to_str_path(path)

    # Files are always copied to a tempfolder to avoid different cases for local/remote
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_tmp_path = os.path.join(tmp_dir, "remote_file.cue")
        with open(local_tmp_path, "w", encoding="utf8") as tmp_f:
            logger.info(f"Copying '{path_str}' to {local_tmp_path} for parsing...")
            with fsspec.open(path_str, "r", encoding="utf8") as f:
                contents = f.read()
                tmp_f.write(contents)
                tmp_f.flush()
            result = load_local(local_tmp_path)
    return result

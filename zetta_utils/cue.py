"""cuelang parsing."""
import json
import pathlib
import os
from subprocess import run

cue_exe = os.environ.get("CUE_EXE", "cue")


def load(cue_file):
    if isinstance(cue_file, (str, pathlib.PosixPath)):
        file_name = str(cue_file)
    elif hasattr(cue_file, "name"):  # File pointers
        file_name = cue_file.name
    else:
        raise ValueError("Invalid input.")

    if not os.path.exists(file_name):
        # CUE will raise an exception here, but it will look for the file as a moudle if
        # the file extension is not correct. We raise manually for more descriptive error
        raise FileNotFoundError(file_name)  # pragma: no cover
    dir_name = os.path.dirname(file_name)
    command_result = run(
        [cue_exe, "export", file_name], capture_output=True, check=False, cwd=dir_name
    )
    if command_result.returncode != 0:
        raise RuntimeError(  # pragma: no cover
            f"CUE failed parsing {file_name}: {command_result.stderr}"
        )
    result_str = command_result.stdout
    result = json.loads(result_str)
    return result

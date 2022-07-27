"""cuelang parsing."""
import json
import pathlib
from os import environ
from subprocess import run

cue_exe = environ.get("CUE_EXE", "cue")


def load(cue_file):
    """Validate from files"""
    if isinstance(cue_file, (str, pathlib.PosixPath)):
        file_name = str(cue_file)
    elif hasattr(cue_file, "name"):  # File pointers
        file_name = cue_file.name
    else:
        raise ValueError("Invalid input.")

    command_result = run([cue_exe, "export", file_name], capture_output=True, check=True)
    result_str = command_result.stdout
    result = json.loads(result_str)
    return result

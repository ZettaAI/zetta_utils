"""cuelang parsing."""
import tempfile
import subprocess
import json
import pathlib
import os
import fsspec  # type: ignore


cue_exe = os.environ.get("CUE_EXE", "cue")


def load(cue_file):
    if isinstance(cue_file, (str, pathlib.PosixPath)):
        path = str(cue_file)
    elif hasattr(cue_file, "name"):  # File pointers
        path = cue_file.name
    else:
        raise ValueError("Invalid input.")

    # Files are always copied to a tempfolder to avoid different cases for local/remote
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_tmp_path = os.path.join(tmp_dir, "remote_file.cue")
        with open(local_tmp_path, "w", encoding="utf8") as tmp_f:
            with fsspec.open(path, "r", encoding="utf8") as f:
                tmp_f.write(f.read())

        command_result = subprocess.run(
            [cue_exe, "export", local_tmp_path], capture_output=True, check=False
        )

    if command_result.returncode != 0:
        raise RuntimeError(  # pragma: no cover
            f"CUE failed parsing {path} (copied to tmp_path '{local_tmp_path}'): "
            f"{command_result.stderr}"
        )

    result_str = command_result.stdout
    result = json.loads(result_str)
    return result

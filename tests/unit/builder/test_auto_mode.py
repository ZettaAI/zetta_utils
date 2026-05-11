# pylint: disable=missing-docstring
"""End-to-end smoke for setup_environment('auto', cue_path=...).

Spawns a subprocess so we get a clean REGISTRY (the conftest autouse fixture
populates 'all' for the main test process). Verifies that:
  - setup_environment('auto') with a real CUE file scans, computes a preload
    set, spawns the daemon with that set, and builds successfully.
  - The computed preload set is small relative to "all".
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

CUE_SPEC = """
{
    "@type": "BBox3D.from_coords"
    start_coord: [0, 0, 0]
    end_coord: [10, 10, 10]
    resolution: [1, 1, 1]
}
"""


@pytest.mark.timeout(120)
def test_auto_mode_builds_via_computed_preload(tmp_path: Path):
    cue_file = tmp_path / "spec.cue"
    cue_file.write_text(CUE_SPEC.strip() + "\n")

    script = textwrap.dedent(
        f"""
        import json
        import zetta_utils
        zetta_utils.setup_environment("auto", cue_path={str(cue_file)!r})

        # pylint: disable=protected-access
        from multiprocessing.forkserver import _forkserver
        preload_used = list(_forkserver._preload_modules)

        from zetta_utils import builder
        from zetta_utils.parsing import cue
        spec = cue.load({str(cue_file)!r})
        result = builder.build(spec=spec)

        print(json.dumps({{
            "preload_used": preload_used,
            "preload_size": len(preload_used),
            "result_repr": repr(result),
        }}))
        """
    ).strip()
    out = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    payload = json.loads(out.stdout.strip().splitlines()[-1])

    assert "BBox3D" in payload["result_repr"]
    # ALWAYS_EAGER (3) + at least geometry.bbox (1) = 4. Far below "all".
    assert 4 <= payload["preload_size"] <= 20, (
        f"auto preload set should be small; got {payload['preload_size']}: "
        f"{payload['preload_used']}"
    )
    assert any(m.endswith("geometry.bbox") for m in payload["preload_used"])
    assert "zetta_utils.builder" in payload["preload_used"]


@pytest.mark.timeout(60)
def test_auto_mode_without_cue_path_falls_back():
    """auto with no cue_path warns and falls back; doesn't crash."""
    script = textwrap.dedent(
        """
        import json
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import zetta_utils
            zetta_utils.setup_environment("auto")  # no cue_path
        msgs = [str(w.message) for w in caught if "auto" in str(w.message).lower()]
        print(json.dumps({"warned": bool(msgs), "sample": msgs[:1]}))
        """
    ).strip()
    out = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(out.stdout.strip().splitlines()[-1])
    assert payload["warned"], f"expected fallback warning; got: {payload}"

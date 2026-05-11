# pylint: disable=missing-docstring
"""End-to-end smoke for setup_environment('none') + lookup-miss fallback.

Spawns a subprocess so we get a clean REGISTRY (the conftest autouse fixture
populates 'all' for the main test process). Verifies that:
  - setup_environment('none') succeeds
  - REGISTRY is sparse afterward (only always-eager registrations)
  - build()ing a spec whose @type is in a non-eager module triggers the
    fallback and produces the expected result
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from zetta_utils.builder.preload import ALWAYS_EAGER


def test_always_eager_set_is_explicit():
    """Sanity: ALWAYS_EAGER is a tuple of dotted module strings, all under zetta_utils."""
    assert isinstance(ALWAYS_EAGER, tuple)
    assert all(isinstance(m, str) for m in ALWAYS_EAGER)
    assert all(m.startswith("zetta_utils.") for m in ALWAYS_EAGER)


def test_preload_none_module_importable():
    """preload.none imports cleanly and pulls in the always-eager set."""
    import importlib  # pylint: disable=import-outside-toplevel

    importlib.import_module("zetta_utils.builder.preload.none")
    importlib.import_module("zetta_utils.builder.preload")


def test_load_mode_literal_includes_none():
    import typing  # pylint: disable=import-outside-toplevel

    from zetta_utils import LoadMode  # pylint: disable=import-outside-toplevel

    assert "none" in typing.get_args(LoadMode)


@pytest.mark.timeout(60)
def test_subprocess_none_mode_uses_fallback_to_build():
    """In a clean subprocess: setup('none') → build a non-eager spec via fallback."""
    script = textwrap.dedent(
        """
        import json
        import zetta_utils
        zetta_utils.setup_environment("none")

        from zetta_utils import builder
        from zetta_utils.builder import registry

        before_size = len(registry.REGISTRY)
        before_attempted = len(registry._lazy_attempted)

        # BBox3D.from_coords lives in zetta_utils.geometry.bbox, not in the
        # always-eager preload. Building it should trigger the lazy fallback.
        spec = {
            "@type": "BBox3D.from_coords",
            "start_coord": [0, 0, 0],
            "end_coord": [10, 10, 10],
            "resolution": [1, 1, 1],
        }
        result = builder.build(spec=spec)

        after_size = len(registry.REGISTRY)
        after_attempted = len(registry._lazy_attempted)

        print(json.dumps({
            "result_repr": repr(result),
            "before_size": before_size,
            "after_size": after_size,
            "before_attempted": before_attempted,
            "after_attempted": after_attempted,
        }))
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

    assert payload["before_attempted"] == 0
    assert payload["after_attempted"] >= 1, "lazy fallback should have fired at least once"
    assert (
        payload["after_size"] > payload["before_size"]
    ), "registry should have grown as fallback imported geometry/bbox"
    assert "BBox3D" in payload["result_repr"]

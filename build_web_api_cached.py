#!/usr/bin/env python3
# pylint: disable=missing-docstring,wrong-import-position
"""Cached variant of build_web_api.py.

Same CLI surface as build_web_api.py — every flag (`--bump`, `--rebuild`,
`--tag`, `--variant`, `--no-build`, `--no-push`, `--no-commit`, `--no-latest`,
`--project`, `--region`, `--repo`) behaves identically. The only differences:

- Uses web_api/cached.Dockerfile and web_api/gpu.cached.Dockerfile, which add
  apt cache mounts and reorder layers so pyproject.toml / source edits do not
  bust the heavy pip wheel layers.
- Builds through `docker buildx build --load` so BuildKit cache mounts
  (apt + pip wheels) are engaged explicitly rather than relying on the
  classic builder's BuildKit fallback.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import build_web_api  # noqa: E402

build_web_api.VARIANTS = {
    "cpu": {"image": "web_api", "dockerfile": "web_api/cached.Dockerfile"},
    "gpu": {"image": "web_api_gpu", "dockerfile": "web_api/gpu.cached.Dockerfile"},
}


def _cached_build_variant(
    variant: str, full_tag: str, project: str, region: str, repo: str
) -> bool:
    ref = build_web_api.image_ref(variant, full_tag, project, region, repo)
    dockerfile = build_web_api.VARIANTS[variant]["dockerfile"]
    cmd = (
        f"docker buildx build --platform linux/amd64 --network=host "
        f"-t {ref} -f {dockerfile} --load ."
    )
    return build_web_api.run_shell(cmd) == 0


build_web_api.build_variant = _cached_build_variant


if __name__ == "__main__":
    sys.exit(build_web_api.main())

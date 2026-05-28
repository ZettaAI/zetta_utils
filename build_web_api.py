#!/usr/bin/env python3
# pylint: disable=missing-docstring,line-too-long
import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VERSION_FILE = REPO_ROOT / "web_api" / "VERSION"
GIT_TAG_PREFIX = "webapi_v"

VARIANTS = {
    "cpu": {"image": "web_api", "dockerfile": "web_api/Dockerfile"},
    "gpu": {"image": "web_api_gpu", "dockerfile": "web_api/gpu.Dockerfile"},
}


def read_version() -> str:
    if not VERSION_FILE.exists():
        raise SystemExit(f"VERSION file not found at {VERSION_FILE}")
    return VERSION_FILE.read_text().strip()


def write_version(version: str) -> None:
    VERSION_FILE.write_text(version + "\n")


def bump_semver(version: str, part: str) -> str:
    major, minor, patch = (int(x) for x in version.split("."))
    if part == "major":
        major, minor, patch = major + 1, 0, 0
    elif part == "minor":
        minor, patch = minor + 1, 0
    else:
        patch += 1
    return f"{major}.{minor}.{patch}"


def next_build_number(semver: str, date_str: str) -> int:
    subprocess.run(["git", "fetch", "--tags"], check=False, capture_output=True)
    result = subprocess.run(
        ["git", "tag", "--list", f"{GIT_TAG_PREFIX}{semver}-{date_str}*"],
        capture_output=True,
        text=True,
        check=True,
    )
    pattern = re.compile(
        rf"^{re.escape(GIT_TAG_PREFIX)}{re.escape(semver)}-{date_str}(\d{{2}})$"
    )
    nums = []
    for line in result.stdout.splitlines():
        m = pattern.match(line.strip())
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


def run_shell(cmd: str) -> int:
    print(f"Running:\n{cmd}")
    return subprocess.call(cmd, shell=True)


def image_ref(variant: str, full_tag: str, project: str, region: str, repo: str) -> str:
    return f"{region}-docker.pkg.dev/{project}/{repo}/{VARIANTS[variant]['image']}:{full_tag}"


def build_variant(variant: str, full_tag: str, project: str, region: str, repo: str) -> bool:
    ref = image_ref(variant, full_tag, project, region, repo)
    dockerfile = VARIANTS[variant]["dockerfile"]
    cmd = (
        f"docker build --platform linux/amd64 --network=host "
        f"-t {ref} -f {dockerfile} ."
    )
    return run_shell(cmd) == 0


def push_variant(variant: str, full_tag: str, project: str, region: str, repo: str) -> bool:
    return run_shell(f"docker push {image_ref(variant, full_tag, project, region, repo)}") == 0


def update_latest(variant: str, full_tag: str, project: str, region: str, repo: str) -> bool:
    src = image_ref(variant, full_tag, project, region, repo)
    dst = image_ref(variant, "latest", project, region, repo)
    if run_shell(f"docker tag {src} {dst}") != 0:
        return False
    return run_shell(f"docker push {dst}") == 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and push web_api docker images (CPU and/or GPU)."
    )
    parser.add_argument("--variant", choices=["cpu", "gpu", "both"], default="both")
    parser.add_argument("--project", default="zetta-research", help="GCR project.")
    parser.add_argument("--region", default="us-east1", help="Artifact Registry region.")
    parser.add_argument("--repo", default="zutils", help="Artifact Registry repo name.")

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--bump",
        choices=["major", "minor", "patch"],
        help="Bump semver in VERSION file. Build number resets to 01 for the day.",
    )
    action.add_argument(
        "--rebuild",
        action="store_true",
        help="Keep current semver from VERSION; increment today's build number.",
    )
    action.add_argument(
        "--tag",
        type=str,
        help="Explicit tag override. Bypasses VERSION logic and skips git tagging.",
    )

    parser.add_argument("--no-build", action="store_true", help="Skip docker build.")
    parser.add_argument(
        "--no-push", action="store_true", help="Skip docker push and git tagging."
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="With --bump: skip auto-commit of VERSION file after successful push.",
    )
    parser.add_argument(
        "--no-latest",
        action="store_true",
        help="Skip tagging and pushing the :latest image (default: push :latest on --bump/--rebuild).",
    )
    return parser


def _resolve_tag(args):
    if args.tag:
        return args.tag, None
    current = read_version()
    new_semver = bump_semver(current, args.bump) if args.bump else current
    date_str = datetime.now().strftime("%Y%m%d")
    build_num = next_build_number(new_semver, date_str)
    full_tag = f"{new_semver}-{date_str}{build_num:02d}"
    if args.bump:
        print(f"Semver bump: {current} -> {new_semver}")
    return full_tag, new_semver


def _run_builds(variants, full_tag, args) -> bool:
    for v in variants:
        if not build_variant(v, full_tag, args.project, args.region, args.repo):
            print(f"Build failed for {v}, exiting.")
            return False
    return True


def _run_pushes(variants, full_tag, args) -> bool:
    for v in variants:
        if not push_variant(v, full_tag, args.project, args.region, args.repo):
            print(f"Push failed for {v}, exiting.")
            return False
    return True


def _run_latest(variants, full_tag, args) -> bool:
    for v in variants:
        if not update_latest(v, full_tag, args.project, args.region, args.repo):
            print(f"Failed to update :latest for {v}, exiting.")
            return False
    return True


def _write_and_commit_version(new_semver, no_commit: bool) -> bool:
    write_version(new_semver)
    print(f"Updated VERSION file to {new_semver}.")
    if no_commit:
        return True
    if subprocess.call(["git", "add", str(VERSION_FILE)]) != 0:
        print("git add failed.")
        return False
    msg = f"chore: bump web_api version to {new_semver}"
    if subprocess.call(["git", "commit", "-m", msg]) != 0:
        print("git commit failed.")
        return False
    return True


def _create_and_push_git_tag(full_tag: str) -> bool:
    git_tag = f"{GIT_TAG_PREFIX}{full_tag}"
    print(f"Creating git tag {git_tag}.")
    if subprocess.call(["git", "tag", git_tag]) != 0:
        print("git tag failed.")
        return False
    if subprocess.call(["git", "push", "origin", git_tag]) != 0:
        print("git tag push failed.")
        return False
    return True


def _run_post_build(variants, full_tag, args, new_semver) -> bool:
    if not _run_pushes(variants, full_tag, args):
        return False
    if not args.tag and not args.no_latest and not _run_latest(variants, full_tag, args):
        return False
    if args.bump and not _write_and_commit_version(new_semver, args.no_commit):
        return False
    if not args.tag and not _create_and_push_git_tag(full_tag):
        return False
    return True


def main() -> int:
    args = _build_parser().parse_args()
    full_tag, new_semver = _resolve_tag(args)
    print(f"Target tag: {full_tag}")
    variants = ["cpu", "gpu"] if args.variant == "both" else [args.variant]
    print(f"Variants: {variants}")
    if not args.no_build and not _run_builds(variants, full_tag, args):
        return 1
    if args.no_push:
        return 0
    if not _run_post_build(variants, full_tag, args, new_semver):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

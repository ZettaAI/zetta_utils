#!/usr/bin/env python3
# pylint: disable=missing-docstring,line-too-long
import argparse
import subprocess

BUILD_COMMAND_TMPL = (
    "docker build --platform linux/amd64 --network=host "
    "-t {REGION}-docker.pkg.dev/{PROJECT}/{REPO}/segment_update_worker:{TAG} "
    "-f zetta_utils/task_management/automated_workers/Dockerfile.segment_update_worker ."
)
PUSH_COMMAND_TMPL = (
    "docker push {REGION}-docker.pkg.dev/{PROJECT}/{REPO}/segment_update_worker:{TAG}"
)


def main():
    parser = argparse.ArgumentParser(
        description="Build and push Skeleton Update Worker docker image."
    )
    parser.add_argument("--project", type=str, required=True, help="GCP project.")
    parser.add_argument("--tag", "-t", type=str, required=True, help="Image tag name/version.")
    parser.add_argument(
        "--region", type=str, default="us-east1", help="Artifact Registry region."
    )
    parser.add_argument(
        "--repo", type=str, default="zutils", help="Artifact Registry repo name."
    )

    args = parser.parse_args()

    build_command = (
        BUILD_COMMAND_TMPL.replace("{TAG}", args.tag)
        .replace("{PROJECT}", args.project)
        .replace("{REGION}", args.region)
        .replace("{REPO}", args.repo)
    )
    print(f"Running: \n{build_command}")
    subprocess.call(build_command, shell=True)

    push_command = (
        PUSH_COMMAND_TMPL.replace("{TAG}", args.tag)
        .replace("{PROJECT}", args.project)
        .replace("{REGION}", args.region)
        .replace("{REPO}", args.repo)
    )
    print(f"Running: \n{push_command}")
    subprocess.call(push_command, shell=True)

    print(f"\nAdding git tag segment_update_worker_{args.tag}.")
    subprocess.call(
        f"git tag segment_update_worker_{args.tag} && git push origin segment_update_worker_{args.tag}",
        shell=True,
    )


if __name__ == "__main__":
    main()

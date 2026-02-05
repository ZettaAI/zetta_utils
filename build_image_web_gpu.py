#!/usr/bin/env python3
# pylint: disable=missing-docstring,line-too-long
import argparse
import subprocess

BUILD_COMMAND_TMPL = "docker build --platform linux/amd64 --network=host -t {REGION}-docker.pkg.dev/{PROJECT}/{REPO}/web_api_gpu:{TAG} -f web_api/Dockerfile.gpu ."
PUSH_COMMAND_TMPL = "docker push {REGION}-docker.pkg.dev/{PROJECT}/{REPO}/web_api_gpu:{TAG}"


def main():
    parser = argparse.ArgumentParser(description="Build and push GPU docker image.")
    parser.add_argument("--project", type=str, required=True, help="GCR project.")
    parser.add_argument("--tag", "-t", type=str, required=True, help="Image tag name/version.")
    parser.add_argument("--region", type=str, default="us-east1", help="Artifact Registry region.")
    parser.add_argument("--repo", type=str, default="zutils", help="Artifact Registry repo name.")

    args = parser.parse_args()

    build_command = BUILD_COMMAND_TMPL
    build_command = build_command.replace("{TAG}", args.tag)
    build_command = build_command.replace("{PROJECT}", args.project)
    build_command = build_command.replace("{REGION}", args.region)
    build_command = build_command.replace("{REPO}", args.repo)
    print(f"Running: \n{build_command}")
    if subprocess.call(build_command, shell=True) != 0:
        print("Build failed, exiting.")
        return

    push_command = PUSH_COMMAND_TMPL
    push_command = push_command.replace("{TAG}", args.tag)
    push_command = push_command.replace("{PROJECT}", args.project)
    push_command = push_command.replace("{REGION}", args.region)
    push_command = push_command.replace("{REPO}", args.repo)
    print(f"Running: \n{push_command}")
    if subprocess.call(push_command, shell=True) != 0:
        print("Push failed, exiting.")
        return

    print(f"\nAdding git tag {args.tag}_gpu.")
    if subprocess.call(f"git tag {args.tag}_gpu && git push origin {args.tag}_gpu", shell=True) != 0:
        print("Git tagging failed, exiting.")
        return


if __name__ == "__main__":
    main()

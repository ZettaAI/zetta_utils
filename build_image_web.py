#!python
# pylint: disable=missing-docstring,line-too-long
import argparse
import subprocess

BUILD_COMMAND_TMPL = "docker build --network=host -t {REGION}-docker.pkg.dev/{PROJECT}/{REPO}/web_api:{TAG} -f web_api/Dockerfile ."
PUSH_COMMAND_TMPL = "docker push {REGION}-docker.pkg.dev/{PROJECT}/{REPO}/web_api:{TAG}"


def main():
    parser = argparse.ArgumentParser(description="Build and push docker image.")
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
    subprocess.call(build_command, shell=True)

    push_command = PUSH_COMMAND_TMPL
    push_command = push_command.replace("{TAG}", args.tag)
    push_command = push_command.replace("{PROJECT}", args.project)
    push_command = push_command.replace("{REGION}", args.region)
    push_command = push_command.replace("{REPO}", args.repo)
    print(f"Running: \n{push_command}")
    subprocess.call(push_command, shell=True)


if __name__ == "__main__":
    main()

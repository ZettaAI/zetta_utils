#!python
# pylint: disable=missing-docstring,line-too-long
import argparse
import subprocess

BUILD_COMMAND_TMPL = "docker build -t us.gcr.io/{PROJECT}/zetta_utils:{TAG} -f docker/Dockerfile.{MODE}.{PYTHON_VERSION} ."
PUSH_COMMAND_TMPL = "docker push us.gcr.io/{PROJECT}/zetta_utils:{TAG}"


def main():
    parser = argparse.ArgumentParser(description="Build and push docker image.")
    parser.add_argument(
        "--project", type=str, required=True, help="GCR project.", default="zetta-research"
    )
    parser.add_argument("--tag", "-t", type=str, required=True, help="Image tag name/version.")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="all",
        choices=["all", "training", "inference"],
        help="Which dependencies to install.",
    )
    parser.add_argument(
        "--python",
        "-p",
        type=str,
        default="3.9",
        choices=["3.9"],
        help="Which python version to use for the image.",
    )

    args = parser.parse_args()

    build_command = BUILD_COMMAND_TMPL
    build_command = build_command.replace("{TAG}", args.tag)
    build_command = build_command.replace("{MODE}", args.mode)
    build_command = build_command.replace("{PYTHON_VERSION}", f"p{args.python.replace('.', '')}")
    build_command = build_command.replace("{PROJECT}", args.project)
    print(f"Running: \n{build_command}")
    subprocess.call(build_command, shell=True)

    push_command = PUSH_COMMAND_TMPL
    push_command = push_command.replace("{TAG}", args.tag)
    push_command = push_command.replace("{PROJECT}", args.project)
    print(f"Running: \n{push_command}")
    subprocess.call(push_command, shell=True)


if __name__ == "__main__":
    main()

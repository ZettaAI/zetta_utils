#!python
# pylint: disable=missing-docstring,line-too-long
import argparse
import subprocess

BUILD_COMMAND_TMPL = (
    "docker build -t us.gcr.io/zetta-research/zetta_utils:{TAG} -f docker/Dockerfile.{MODE} ."
)
PUSH_COMMAND_TMPL = "docker push us.gcr.io/zetta-research/zetta_utils:{TAG}"


def main():
    parser = argparse.ArgumentParser(description="Build and push docker image.")
    parser.add_argument("--tag", "-t", type=str, required=True, help="Image tag name/version.")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="all",
        choices=["all", "training", "inference"],
        help="Which dependencies to install.",
    )

    args = parser.parse_args()

    build_command = BUILD_COMMAND_TMPL
    build_command = build_command.replace("{TAG}", args.tag)
    build_command = build_command.replace("{MODE}", args.mode)
    print(f"Running: \n{build_command}")
    subprocess.call(build_command, shell=True)

    push_command = PUSH_COMMAND_TMPL
    push_command = push_command.replace("{TAG}", args.tag)
    print(f"Running: \n{push_command}")
    subprocess.call(push_command, shell=True)


if __name__ == "__main__":
    main()

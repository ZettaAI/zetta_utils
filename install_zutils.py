#!/usr/bin/env python3
# pylint: disable=wrong-import-position,line-too-long,too-many-branches,too-many-statements

import argparse
import os
import platform
import subprocess
import sys
import threading
from typing import Any


def ensure_dependencies():
    """Ensure all required dependencies are available, install if they aren't."""
    dependencies = {"rich": "rich", "prompt_toolkit": "prompt_toolkit"}

    installed = []
    for module, package in dependencies.items():
        try:
            __import__(module)
        except ImportError:
            print(f"Installing required dependency: {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                installed.append(package)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Failed to install {package}: {e}")
                sys.exit(1)

    if installed:
        print("Required dependencies have been installed. Restarting script...")
        os.execv(sys.executable, ["python"] + sys.argv)


# Ensure non-standard dependencies are available before proceeding
ensure_dependencies()

from prompt_toolkit import PromptSession
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "command": "bright_black",
    }
)

console = Console(theme=custom_theme)


def print_error(message):
    console.print(Panel(message, style="error", title="Error"))
    sys.exit(1)


def print_warning(message):
    console.print(Panel(message, style="warning", title="Warning"))


def print_success(message):
    console.print(Panel(message, style="success", title="Success"))


def run_command(command, description=None, interactive=False):
    """Execute a command and capture its output while showing a status spinner."""
    status_message = f"[info]{description or command}..."

    if interactive:
        # For interactive commands, use a simpler approach that preserves stdin/stdout
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError:
            print_error(f"Error executing command: {command}")
            raise
        return

    with console.status(status_message) as status:
        try:
            with subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            ) as process:
                # Use separate threads to read stdout and stderr
                def read_output(pipe, style):
                    for line in iter(pipe.readline, ""):
                        if line.strip():
                            status.stop()
                            escaped_line = line.strip().replace("[", "\\[").replace("]", "\\]")
                            console.print(f"[{style}]{escaped_line}[/{style}]")
                            status.start()

                stdout_thread = threading.Thread(
                    target=read_output, args=(process.stdout, "command")
                )
                stderr_thread = threading.Thread(
                    target=read_output, args=(process.stderr, "error")
                )

                stdout_thread.daemon = True
                stderr_thread.daemon = True

                stdout_thread.start()
                stderr_thread.start()

                return_code = process.wait()

                stdout_thread.join()
                stderr_thread.join()

                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, command)

        except subprocess.CalledProcessError:
            print_error(f"Error executing command: {command}")
            raise


def check_ubuntu():
    with console.status("[info]Checking system compatibility...", spinner="dots"):
        if platform.system() != "Linux":
            print_error(
                f"This script must be run on a Linux system.\nCurrent system detected: {platform.system()}"
            )

        try:
            with open("/etc/os-release", "r", encoding="utf8") as f:
                os_info = dict(
                    line.strip().split("=", 1)
                    for line in f
                    if "=" in line and not line.startswith("#")
                )

            if "ubuntu" not in os_info.get("ID", "").lower():
                print_error(
                    f"This script must be run on Ubuntu.\n"
                    f"Current system detected: {os_info.get('PRETTY_NAME', 'Unknown')}\n\n"
                    "Please modify the script for your specific distribution or use Ubuntu."
                )

        except FileNotFoundError:
            print_error(
                "Cannot determine Linux distribution.\n"
                "This script is designed for Ubuntu systems."
            )


def check_repository():
    with console.status("[info]Verifying repository...", spinner="dots"):
        if not os.path.exists(".git"):
            print_error(
                "This script must be run from the root of zetta_utils repository.\nCould not find .git directory."
            )

        try:
            remote_url = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"], text=True
            ).strip()

            expected_urls = [
                "https://github.com/ZettaAI/zetta_utils.git",
                "https://github.com/ZettaAI/zetta_utils",
                "git@github.com:ZettaAI/zetta_utils.git",
            ]

            if not any(url in remote_url for url in expected_urls):
                print_error(
                    f"This script must be run from the root of zetta_utils repository.\n"
                    f"Current repository: {remote_url}\n"
                    "Expected repository: https://github.com/ZettaAI/zetta_utils"
                )

        except subprocess.CalledProcessError:
            print_error(
                "Failed to get git remote URL.\n"
                "Please ensure you're in the zetta_utils repository root."
            )


def get_current_shell():
    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        return "zsh", os.path.expanduser("~/.zshrc")
    elif "bash" in shell:
        return "bash", os.path.expanduser("~/.bashrc")
    else:
        print_warning(f"Unrecognized shell {shell}\nDefaulting to bash")
        return "bash", os.path.expanduser("~/.bashrc")


def get_input(prompt, session, validator=None, default=None, show_default=True):
    display_prompt = prompt
    if default and show_default:
        display_prompt = f"{prompt} [{default}]"

    while True:
        try:
            value = session.prompt(display_prompt + ": ")
            value = value.strip()
            if not value and default:
                value = default
            if validator:
                value = validator(value)
            return value
        except ValueError as e:
            console.print(f"[error]{str(e)}[/error]")


def validate_aws_region(value):
    """Validate AWS region."""
    valid_regions = [
        "us-east-1",
        "us-east-2",
        "us-west-1",
        "us-west-2",
        "eu-west-1",
        "eu-west-2",
        "eu-central-1",
        "ap-southeast-1",
        "ap-southeast-2",
        "ap-northeast-1",
    ]
    if not value or value.isspace():
        raise ValueError("AWS region cannot be empty")
    if value not in valid_regions:
        raise ValueError(f"Invalid AWS region. Must be one of: {', '.join(valid_regions)}")
    return value


def validate_aws_access_key(value):
    """Validate AWS access key ID."""
    if not value or value.isspace():
        raise ValueError("AWS Access Key ID cannot be empty")
    if not value.startswith("AKI"):
        raise ValueError("AWS Access Key ID should start with 'AKI'")
    if len(value) != 20:
        raise ValueError("AWS Access Key ID should be 20 characters long")
    return value


def validate_aws_secret_key(value):
    """Validate AWS secret access key."""
    if not value or value.isspace():
        raise ValueError("AWS Secret Access Key cannot be empty")
    if len(value) != 40:
        raise ValueError("AWS Secret Access Key should be 40 characters long")
    return value


def validate_username(value):
    """Validate Zetta username."""
    if not value or value.isspace():
        raise ValueError("Username cannot be empty")
    if len(value) < 3:
        raise ValueError("Username must be at least 3 characters")
    return value


def validate_project(value):
    """Validate Zetta project name."""
    if not value or value.isspace():
        raise ValueError("Project name cannot be empty")
    if len(value) < 2:
        raise ValueError("Project name must be at least 2 characters")
    return value


ZETTA_ENV_VARS = {
    "ZETTA_USER": {
        "prompt": "Enter your Zetta username",
        "validator": validate_username,
        "default": os.environ.get("ZETTA_USER", ""),
        "show_default": True,
        "sensitive": False,
    },
    "ZETTA_PROJECT": {
        "prompt": "Enter your Zetta project name",
        "validator": validate_project,
        "default": os.environ.get("ZETTA_PROJECT", ""),
        "show_default": True,
        "sensitive": False,
    },
}

AWS_ENV_VARS = {
    "AWS_DEFAULT_REGION": {
        "prompt": "Enter AWS Default Region",
        "validator": validate_aws_region,
        "default": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        "show_default": True,
        "sensitive": False,
    },
    "AWS_ACCESS_KEY_ID": {
        "prompt": "Enter AWS Access Key ID",
        "validator": validate_aws_access_key,
        "default": os.environ.get("AWS_ACCESS_KEY_ID", ""),
        "show_default": False,
        "sensitive": False,
    },
    "AWS_SECRET_ACCESS_KEY": {
        "prompt": "Enter AWS Secret Access Key",
        "validator": validate_aws_secret_key,
        "default": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        "show_default": False,
        "sensitive": True,
    },
}


def write_env_vars_to_rc(vars_to_set, rc_file):
    """Write environment variables to shell RC file."""
    try:
        with open(rc_file, "a", encoding="utf8") as f:
            f.write("\n# Zetta Utils Environment Variables\n")
            for var, value in vars_to_set.items():
                f.write(f'export {var}="{value}"\n')
        return True
    except Exception as e:  # pylint: disable=broad-exception-caught
        print_error(f"Error writing to {rc_file}:\n{str(e)}")
        return False


def display_env_vars(env_vars):
    """Display current environment variables in a table."""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Variable")
    table.add_column("Value")

    for var, config in env_vars.items():
        value = os.environ.get(var, "")
        displayed_value = "*" * len(value) if config["sensitive"] else value
        table.add_row(var, displayed_value)

    console.print(table)


def check_and_setup_zetta_env():
    """Check and set up environment variables."""
    _, rc_file = get_current_shell()
    vars_to_set = {}
    session: Any = PromptSession()

    for var, config in ZETTA_ENV_VARS.items():
        if not os.environ.get(var):
            try:
                value = get_input(
                    prompt=config["prompt"],
                    validator=config["validator"],
                    default=config["default"],
                    show_default=config["show_default"],
                    session=session,
                )
                vars_to_set[var] = value
            except (KeyboardInterrupt, EOFError):
                console.print("\n[error]Setup cancelled by user[/error]")
                sys.exit(1)
        else:
            console.print(f"[info]Skipping {var} as it's already set[/info]")

    while True:
        console.print(
            "\n[cyan]Are you absolutely certain you will never need to run jobs on remote GCP clusters from this machine? [white](y/N)[/cyan]"
        )
        response = session.prompt("").lower()
        if response in ["y", "yes"]:
            no_aws = True
            break
        if response in ["n", "no", ""]:  # empty input defaults to No
            no_aws = False
            break
        console.print(
            f"[yellow]Please enter 'y' for yes or 'n' for no (or press Enter for no), you entered: {response}[/yellow]"
        )
        continue

    if not no_aws:
        console.print(
            "\n[info]Since you're not certain, we'll set up AWS credentials to ensure you're prepared.[/info]"
        )
        console.print(
            "[yellow]Note: If you don't know how to get AWS credentials, please message the team on Slack![/yellow]\n"
        )

        for var, config in AWS_ENV_VARS.items():
            if not os.environ.get(var):
                try:
                    value = get_input(
                        prompt=config["prompt"],
                        validator=config["validator"],
                        default=config["default"],
                        show_default=config["show_default"],
                        session=session,
                    )
                    vars_to_set[var] = value
                except (KeyboardInterrupt, EOFError):
                    console.print("\n[error]Setup cancelled by user[/error]")
                    sys.exit(1)
            else:
                console.print(f"[info]Skipping {var} as it's already set[/info]")

    while True:
        console.print(
            "\n[cyan]Are you absolutely certain you will never need to train models on this machine? [white](y/N)[/cyan]"
        )
        response = session.prompt("").lower()
        if response in ["y", "yes"]:
            no_training = True
            break
        if response in ["n", "no", ""]:  # empty input defaults to No
            no_training = False
            break
        console.print(
            "[yellow]Please enter 'y' for yes or 'n' for no (or press Enter for no)[/yellow]"
        )
        continue

    if not no_training and not os.environ.get("WANDB_API_KEY"):
        console.print(
            "\n[info]Since you're not certain, we'll set up training credentials to ensure you're prepared.[/info]"
        )
        console.print(
            "[yellow]Note: You'll need a Weights & Biases API key for training.[/yellow]"
        )
        console.print(
            "[yellow]You can find your key here: https://docs.wandb.ai/support/find_api_key/[/yellow]\n"
        )

        def validate_wandb_key(value):
            """Validate WANDB API key."""
            if not value or value.isspace():
                raise ValueError("WANDB API key cannot be empty")
            if len(value) != 40:
                raise ValueError("WANDB API key should be 40 characters long")
            return value

        try:
            value = get_input(
                prompt="Enter your Weights & Biases (wandb) API key",
                validator=validate_wandb_key,
                default="",
                show_default=False,
                session=session,
            )
            vars_to_set["WANDB_API_KEY"] = value
        except (KeyboardInterrupt, EOFError):
            console.print("\n[error]Setup cancelled by user[/error]")
            sys.exit(1)
    elif not no_training:
        console.print("[info]Skipping WANDB_API_KEY as it's already set[/info]")

    if vars_to_set:
        with console.status(f"[info]Adding environment variables to {rc_file}...", spinner="dots"):
            if write_env_vars_to_rc(vars_to_set, rc_file):
                for var, value in vars_to_set.items():
                    os.environ[var] = value

                print_success(
                    "Environment variables have been configured:\n"
                    + "\n".join(
                        f"  {var}={'*' * len(value) if var in ['AWS_SECRET_ACCESS_KEY', 'WANDB_API_KEY'] else value}"
                        for var, value in vars_to_set.items()
                    )
                    + f"\n\nVariables have been added to {rc_file}\n"
                    + f"Please run 'source {rc_file}' after this script completes\n"
                    + "or start a new terminal session for the changes to take effect."
                )
    else:
        console.print("\n[info]All required environment variables are already set.[/info]")


def check_and_setup_gcp():
    credentials_path = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")

    if not os.path.exists(credentials_path):
        console.print("\n[info]GCP application default credentials not found. Setting up...")
        try:
            run_command(
                "gcloud auth application-default login --quiet",
                "Setting up GCP credentials",
                interactive=True,
            )
            print_success("GCP credentials setup completed successfully!")
        except Exception as e:  # pylint: disable=broad-exception-caught
            print_error(
                f"Error setting up GCP credentials:\n{e}\n\n"
                "Please ensure gcloud CLI is installed and try again."
            )
    else:
        console.print("\n[info]GCP application default credentials already set up.")


def setup_paths():
    """Set up ~/.local/bin in PATH and current directory in PYTHONPATH for both current env and shell config."""
    # Setup for ~/.local/bin
    local_bin = os.path.expanduser("~/.local/bin")
    os.makedirs(local_bin, exist_ok=True)

    # Get current directory
    current_dir = os.getcwd()

    # Add to current environment
    current_path = os.environ.get("PATH", "")
    if local_bin not in current_path:
        os.environ["PATH"] = f"{local_bin}:{current_path}"

    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if current_dir not in current_pythonpath:
        os.environ["PYTHONPATH"] = (
            f"{current_dir}:{current_pythonpath}" if current_pythonpath else current_dir
        )

    # Prepare exports for RC file
    _, rc_file = get_current_shell()
    exports = [
        "# Add local bin to PATH",
        'export PATH="$HOME/.local/bin:$PATH"',
        "",
        "# Add zetta_utils directory to PYTHONPATH",
        f'export PYTHONPATH="{current_dir}:$PYTHONPATH"',
    ]

    try:
        # Read existing content
        with open(rc_file, "r", encoding="utf8") as f:
            content = f.read()

        # Check what needs to be added
        needs_local_bin = 'export PATH="$HOME/.local/bin:$PATH"' not in content
        needs_pythonpath = f'export PYTHONPATH="{current_dir}' not in content

        # Add necessary exports
        if needs_local_bin or needs_pythonpath:
            with open(rc_file, "a", encoding="utf8") as f:
                f.write("\n")
                if needs_local_bin:
                    f.write(f"{exports[0]}\n{exports[1]}\n")
                    console.print(f"[info]Added ~/.local/bin to PATH in {rc_file}[/info]")
                if needs_pythonpath:
                    f.write(f"{exports[3]}\n{exports[4]}\n")
                    console.print(
                        f"[info]Added current directory to PYTHONPATH in {rc_file}[/info]"
                    )
    except Exception as e:  # pylint: disable=broad-exception-caught
        print_warning(f"Could not modify {rc_file}: {e}")


def check_submodules():
    """Check if git submodules are properly initialized."""
    try:
        # Get list of submodules
        submodules = subprocess.check_output(["git", "submodule", "status"], text=True).strip()

        if not submodules:
            return  # No submodules to check

        # Check for uninitialized submodules (those starting with -)
        uninitialized = [
            line.strip() for line in submodules.split("\n") if line.strip().startswith("-")
        ]

        if uninitialized:
            print_error(
                "Git submodules are not initialized.\n"
                "Please clone the repository with --recurse-submodules or run:\n"
                "git submodule update --init --recursive\n\n"
                "Alternatively, use --no_submodules to skip this check."
            )

    except subprocess.CalledProcessError:
        print_error(
            "Failed to check git submodules.\n"
            "Please ensure you're in the zetta_utils repository root."
        )


def main():
    console.print(Panel.fit("Zetta Utils Installer", style="bold cyan", subtitle="v1.0"))

    parser = argparse.ArgumentParser(description="Install zetta_utils and dependencies")
    parser.add_argument(
        "--mode",
        choices=["modules", "all"],
        default="all",
        help='Installation type: "modules" for basic installation or "all" for full installation',
    )
    parser.add_argument(
        "--dockerfile",
        action="store_true",
        default=False,
        help="Whether command is being evoked in a Dockerfile",
    )
    parser.add_argument(
        "--skip_submodules", action="store_true", default=False, help="Skip submodules check"
    )
    parser.add_argument(
        "--skip_apt", action="store_true", default=False, help="Skip apt modules install"
    )
    parser.add_argument(
        "--skip_pip", action="store_true", default=False, help="Skip pip modules install"
    )
    parser.add_argument(
        "--pcg", action="store_true", default=False, help="Whether or not to install PCG"
    )
    parser.add_argument("--pcgtag", default="v2.18.3", help="PCG repo tag to use")

    args = parser.parse_args()

    if not args.dockerfile:
        check_ubuntu()
        check_repository()
        if not args.skip_submodules:
            check_submodules()
        sudo_prefix = "sudo "
    else:
        sudo_prefix = ""

    if not args.skip_apt:
        run_command(f"{sudo_prefix}apt-get update", "Updating package lists")
        run_command(
            command=f"{sudo_prefix}apt-get install -y wget gnupg software-properties-common && "
            "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && "
            f"{sudo_prefix}mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && "
            f"{sudo_prefix}apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub && "
            f"{sudo_prefix}apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && "
            f'{sudo_prefix}add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"',
            description="Adding NVIDIA apt repository",
        )

        run_command(f"{sudo_prefix}apt-get update", "Updating package lists")
        apt_packages = [
            "git",
            "build-essential",
            "ffmpeg",  # opencv
            "libsm6",  # opencv
            "libxext6",  # opencv
            "curl",  # abiss
            "cmake",  # abiss
            "ninja-build",  # abiss
            "pkg-config",  # abiss
            "zstd",  # abiss
            "parallel",  # abiss
            "coreutils",  # abiss
            "libboost-system-dev",  # abiss
            "libboost-iostreams-dev",  # abiss
            "libjemalloc-dev",  # abiss
            "libtbb-dev",  # abiss
            "libboost-dev",  # waterz
            "unixodbc-dev",  # ???
            "cuda-nvrtc-12-0",  # torch jit
        ]
        apt_install_flags = '-y --no-install-recommends -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"'
        run_command(
            f"{sudo_prefix} apt-get install {apt_install_flags} {' '.join(apt_packages)}",
            "Installing required packages",
        )

        run_command(
            "mkdir -p ~/.zetta_utils/cue && "
            "cd ~/.zetta_utils/cue && "
            "wget https://github.com/cue-lang/cue/releases/download/v0.11.1/cue_v0.11.1_linux_amd64.tar.gz &&"
            "tar -xzvf cue_v0.11.1_linux_amd64.tar.gz && "
            f"{sudo_prefix}mv cue /bin/ && "
            "rm -rf ~/.zetta_utils/cue && "
            "cd -",
            "Installing CUE",
        )

        run_command(
            "curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash",
            "Installing Helm"
        )

    if not args.skip_pip:
        install_mode = args.mode
        if args.pcg:
            install_mode = f"{install_mode},pcg"
        run_command(
            f"pip install --upgrade .[{install_mode}]", "Installing `zetta_utils` python package"
        )
        if args.pcg:
            run_command(
                f"pip install --no-deps git+https://github.com/CAVEconnectome/PyChunkedGraph.git@{args.pcgtag}",
                "Install PCG package (do deps)",
            )

    if not args.dockerfile:
        setup_paths()
        check_and_setup_gcp()
        check_and_setup_zetta_env()
        for region in ["us-central1", "us-east1", "us-west1"]:
            run_command(
                f"gcloud auth configure-docker --quiet {region}-docker.pkg.dev",
                "Setting up GCP artifact repository",
            )

        _, rc_file = get_current_shell()
        print_success(
            "Installation completed successfully!\n\n"
            "[red]IMPORTANT:[/red] Environment variables have been updated in this script, but for the best experience,\n"
            "please run the following command to ensure all changes are properly loaded:\n\n"
            f"    [red]source {rc_file}[/red]\n\n"
            "This will load all environment variables and PATH updates for:\n"
            "  - Zetta Utils configuration\n"
            "  - Python package paths\n"
            "  - Local binary paths\n"
            "  - AWS credentials (if configured)\n"
            "  - Wandb credentials (if configured)"
        )


if __name__ == "__main__":
    main()

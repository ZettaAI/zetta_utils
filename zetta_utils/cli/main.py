import os
import pprint
import subprocess
import sys
from tempfile import NamedTemporaryFile
from typing import Optional

import click

import zetta_utils
from zetta_utils import log
from zetta_utils.cli.run.cli import run_info_cli
from zetta_utils.cli.run.cli_update import run_update_cli
from zetta_utils.cli.task_mgmt import task_mgmt

logger = log.get_logger("zetta_utils")


@click.group()
@click.option("-v", "--verbose", count=True, default=2)
@click.option(
    "--load_mode", "-l", type=click.Choice(["all", "inference", "training", "try"]), default="all"
)
def cli(verbose, load_mode):  # pragma: no cover # no logic, delegation
    verbosity_map = {
        1: "WARN",
        2: "INFO",
        3: "DEBUG",
    }

    verbose = min(verbose, 3)
    log.set_verbosity(verbosity_map[verbose])
    log.configure_logger()

    # Save load_mode in the click context so subcommands can access it
    ctx = click.get_current_context()
    ctx.obj = {"load_mode": load_mode}


def validate_py_path(ctx, param, value):  # pylint: disable=unused-argument
    for path in value:
        if not path.endswith(".py"):
            raise click.BadParameter("File must end with .py")
    return value


@cli.command()
@click.argument("path", type=click.Path(), required=False)
@click.option(
    "--str_spec",
    "-s",
    type=str,
    help="Builder specification provided as a string. Must be provided iff "
    "the `path` argument is not given.",
)
@click.option(
    "--run_id",
    "-r",
    type=str,
    help="Provide a current `run_id` for auxiliary processes. Can also be "
    "used to pass a custom `run_id` and prevent random id generation.",
)
@click.option(
    "--pdb",
    "-d",
    type=bool,
    is_flag=True,
    help="When set to `True`, will insert a breakpoint after building.",
)
@click.option(
    "--parallel_builder",
    "-p",
    type=bool,
    is_flag=True,
    help="Whether to pass `parallel` flag to builder.",
)
@click.option(
    "--extra_import",
    "-i",
    "extra_imports",
    type=str,
    multiple=True,
    callback=validate_py_path,
    help="Specify additional imports. Must end with `.py`.",
)
@click.option(
    "--main-run-process/--no-main-run-process",
    type=bool,
    show_default=True,
    default=True,
    is_flag=True,
    help="Enable/disable heartbeat. Disable with caution.",
)
def run(
    path: Optional[str],
    str_spec: Optional[str],
    run_id: Optional[str],
    pdb: bool,
    parallel_builder: bool,
    extra_imports: tuple[str],
    main_run_process: bool,
):
    """Perform ``zetta_utils.builder.build`` action on file contents."""
    ctx = click.get_current_context()
    load_mode = ctx.obj.get("load_mode", "all") if ctx and ctx.obj else "all"

    # Load modules first
    if load_mode == "all":
        zetta_utils.load_all_modules()
    elif load_mode == "inference":  # pragma: no cover
        zetta_utils.load_inference_modules()
    elif load_mode == "try":  # pragma: no cover
        zetta_utils.try_load_train_inference()
    else:  # pragma: no cover
        assert load_mode == "training"
        zetta_utils.load_training_modules()

    from zetta_utils import builder, parsing  # pylint: disable=import-outside-toplevel
    from zetta_utils.run import (  # pylint: disable=import-outside-toplevel
        run_ctx_manager,
    )

    if path is not None:
        assert str_spec is None, "Exactly one of `path` and `str_spec` must be provided."
        try:
            spec = parsing.cue.load(path)
        except subprocess.CalledProcessError as err:  # pragma: no cover
            logger.error("Aborting due to CUE validation failure.")
            sys.exit(err.returncode)
        os.environ["ZETTA_RUN_SPEC_PATH"] = path
    else:
        assert str_spec is not None, "Exactly one of `path` and `str_spec` must be provided."
        spec = parsing.cue.loads(str_spec)
        with NamedTemporaryFile("w", encoding="utf8", delete=False) as f:
            f.write(str_spec)
            os.environ["ZETTA_RUN_SPEC_PATH"] = f.name

    for import_path in extra_imports:
        assert import_path.endswith(".py")
        with open(import_path, "r", encoding="utf-8") as f:
            code = f.read()
            exec(code)  # pylint: disable=exec-used

    if parallel_builder:
        builder.PARALLEL_BUILD_ALLOWED = True

    with run_ctx_manager(spec=spec, run_id=run_id, main_run_process=main_run_process):
        result = builder.build(spec, parallel=parallel_builder)
        logger.debug(f"Outcome: {pprint.pformat(result, indent=4)}")
        if pdb:
            breakpoint()  # pylint: disable=forgotten-debug-statement # pragma: no cover


@cli.command()
def show_registry():
    """Display builder registry."""
    from zetta_utils import builder  # pylint: disable=import-outside-toplevel

    logger.critical(pprint.pformat(builder.REGISTRY, indent=4))


for cmd in run_info_cli.commands.values():
    cli.add_command(cmd)


for cmd in run_update_cli.commands.values():
    cli.add_command(cmd)

# Add task management commands
cli.add_command(task_mgmt)

import os
import pprint
import subprocess
import sys
from tempfile import NamedTemporaryFile
from typing import Optional, cast

import click

from zetta_utils import LoadMode, log, setup_environment
from zetta_utils.cli.run.cli import run_info_cli
from zetta_utils.cli.run.cli_pester_nodes import pester_nodes_cli
from zetta_utils.cli.run.cli_update import run_update_cli
from zetta_utils.cli.task_mgmt import task_mgmt

logger = log.get_logger("zetta_utils")


@click.group()
@click.option("-v", "--verbose", count=True, default=2)
@click.option(
    "--load_mode",
    "-l",
    type=click.Choice(["all", "inference", "training", "try", "none", "auto"]),
    default="auto",
    help="Preload mode (default 'auto'). 'auto' scans the CUE spec and "
    "preloads exactly the modules it needs. 'none' skips eager imports and "
    "resolves @types lazily via the static registry index (fast dev "
    "iteration; slower first-use). 'all' falls back to preloading the full "
    "module set.",
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
    load_mode = cast(LoadMode, ctx.obj.get("load_mode", "auto") if ctx and ctx.obj else "auto")

    # Auto mode needs a CUE file path to scan. When --str_spec is used,
    # materialize it to a tempfile up front so auto can see it.
    auto_cue_path: Optional[str] = path
    if auto_cue_path is None and str_spec is not None and load_mode == "auto":
        with NamedTemporaryFile("w", encoding="utf8", delete=False, suffix=".cue") as f:
            f.write(str_spec)
            auto_cue_path = f.name

    # Remote workers receive the auto preload list via ZETTA_PRELOAD_MODULES so
    # they don't need to re-parse the original CUE. When set it overrides
    # --load_mode and forces auto with the explicit list.
    preload_env = os.environ.get("ZETTA_PRELOAD_MODULES")
    preload_modules = preload_env.split(",") if preload_env else None
    if preload_modules:
        load_mode = cast(LoadMode, "auto")

    setup_environment(
        load_mode,
        cue_path=auto_cue_path if load_mode == "auto" else None,
        preload_modules=preload_modules,
    )

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
        if auto_cue_path is not None:
            os.environ["ZETTA_RUN_SPEC_PATH"] = auto_cue_path
        else:
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


for cmd in pester_nodes_cli.commands.values():
    cli.add_command(cmd)

# Add task management commands
cli.add_command(task_mgmt)

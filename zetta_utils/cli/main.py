import os
import pprint
from typing import Optional

import click

import zetta_utils
from zetta_utils import log
from zetta_utils.parsing import json

logger = log.get_logger("zetta_utils")


@click.group()
@click.option("-v", "--verbose", count=True, default=2)
@click.option(
    "--load_mode", "-l", type=click.Choice(["all", "inference", "training", "try"]), default="all"
)
def cli(load_mode, verbose):  # pragma: no cover # no logic, delegation
    if load_mode == "all":
        zetta_utils.load_all_modules()
    elif load_mode == "inference":
        zetta_utils.load_inference_modules()
    elif load_mode == "try":
        zetta_utils.try_load_train_inference()
    else:
        assert load_mode == "training"
        zetta_utils.load_training_modules()
    verbosity_map = {
        1: "WARN",
        2: "INFO",
        3: "DEBUG",
    }

    verbose = min(verbose, 3)
    log.set_verbosity(verbosity_map[verbose])
    log.configure_logger()


def validate_py_path(ctx, param, value):  # pylint: disable=unused-argument
    for path in value:
        if not path.endswith(".py"):
            raise click.BadParameter("File must end with .py")
    return value


@click.command()
@click.argument("path", type=click.Path(), required=False)
@click.option(
    "--str_spec",
    "-s",
    type=str,
    help="Builder specification provided as a string. Must be provided iff "
    "the `path` argument is not given.",
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
def run(
    path: Optional[str],
    str_spec: Optional[str],
    pdb: bool,
    parallel_builder: bool,
    extra_imports: tuple[str],
):
    """Perform ``zetta_utils.builder.build`` action on file contents."""
    if path is not None:
        assert str_spec is None, "Exactly one of `path` and `str_spec` must be provided."
        spec = zetta_utils.parsing.cue.load(path)
        os.environ["ZETTA_RUN_SPEC_PATH"] = path
    else:
        assert str_spec is not None, "Exactly one of `path` and `str_spec` must be provided."
        spec = zetta_utils.parsing.cue.loads(str_spec)

    os.environ["ZETTA_RUN_SPEC"] = json.dumps(spec)

    for import_path in extra_imports:
        assert import_path.endswith(".py")
        with open(import_path, "r", encoding="utf-8") as f:
            code = f.read()
            exec(code)  # pylint: disable=exec-used

    if parallel_builder:
        zetta_utils.builder.PARALLEL_BUILD_BY_DEFAULT = True

    result = zetta_utils.builder.build(spec, parallel=parallel_builder)
    logger.debug(f"Outcome: {pprint.pformat(result, indent=4)}")
    if pdb:
        breakpoint()  # pylint: disable=forgotten-debug-statement # pragma: no cover


@click.command()
def show_registry():
    """Display builder registry."""
    logger.critical(pprint.pformat(zetta_utils.builder.REGISTRY, indent=4))


cli.add_command(run)
cli.add_command(show_registry)

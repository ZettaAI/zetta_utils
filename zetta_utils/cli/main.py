import json
import os
import pprint
from typing import Optional

import click

import zetta_utils
from zetta_utils import log

logger = log.get_logger("zetta_utils")


@click.group()
@click.option("-v", "--verbose", count=True)
@click.option(
    "--load_mode", "-l", type=click.Choice(["all", "inference", "training"]), default="all"
)
def cli(load_mode, verbose):  # pragma: no cover # no logic, delegation
    if load_mode == "all":
        zetta_utils.load_all_modules()
    elif load_mode == "inference":
        zetta_utils.load_inference_modules()
    else:
        assert load_mode == "training"
        zetta_utils.load_training_modules()

    verbosity_map = {
        0: "WARN",
        1: "INFO",
        2: "DEBUG",
    }

    verbose = min(verbose, 2)
    log.set_verbosity(verbosity_map[verbose])
    log.configure_logger()


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
def run(path: Optional[str], str_spec: Optional[str], pdb: bool):
    """Perform ``zetta_utils.builder.build`` action on file contents."""
    if path is not None:
        assert str_spec is None, "Exectly one of `path` and `str_spec` must be provided."
        spec = zetta_utils.parsing.cue.load(path)
    else:
        assert str_spec is not None, "Exectly one of `path` and `str_spec` must be provided."
        spec = zetta_utils.parsing.cue.loads(str_spec)

    os.environ["ZETTA_RUN_SPEC"] = json.dumps(spec)
    result = zetta_utils.builder.build(spec)
    logger.info(f"Outcome: {pprint.pformat(result, indent=4)}")
    if pdb:
        breakpoint()  # pylint: disable=forgotten-debug-statement # pragma: no cover


@click.command()
def show_registry():
    """Display builder registry."""
    logger.critical(pprint.pformat(zetta_utils.builder.REGISTRY, indent=4))


cli.add_command(run)
cli.add_command(show_registry)

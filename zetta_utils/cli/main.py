import json
import os
import pprint

import click

import zetta_utils
from zetta_utils import log

logger = log.get_logger("zetta_utils")

# For now, CLI requires all modules to be installed
# If the need arises, the installed modules can be specified
# through a config file.
zetta_utils.load_all_modules()


@click.group()
@click.option("-v", "--verbose", count=True)
def cli(verbose):  # pragma: no cover # no logic, delegation
    verbosity_map = {
        0: "WARN",
        1: "INFO",
        2: "DEBUG",
    }

    verbose = min(verbose, 2)
    for k in ["zetta_user", "zetta_project"]:
        assert k.upper() in os.environ, f"Env variable '{k.upper()}' must be set to run zetta cli"
        log.set_logging_label(k, os.environ[k.upper()])

    log.set_verbosity(verbosity_map[verbose])
    log.configure_logger()


@click.command()
@click.argument("path", type=click.Path())
@click.option(
    "--pdb",
    "-d",
    type=bool,
    is_flag=True,
    help="When set to `True`, will insert a breakpoint after building.",
)
def run(path, pdb):
    """Perform ``zetta_utils.builder.build`` action on file contents."""
    spec = zetta_utils.parsing.cue.load(path)
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

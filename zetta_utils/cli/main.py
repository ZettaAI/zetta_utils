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
def cli(verbose):
    log.configure_logger("zetta_utils", level=verbose)  # pragma: no cover


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
    result = zetta_utils.builder.build(spec)
    pprint.pprint(result)
    if pdb:
        breakpoint()  # pylint: disable=forgotten-debug-statement # pragma: no cover


@click.command()
def show_registry():
    """Display builder registry."""
    logger.critical(pprint.pformat(zetta_utils.builder.REGISTRY, indent=4))


cli.add_command(run)
cli.add_command(show_registry)

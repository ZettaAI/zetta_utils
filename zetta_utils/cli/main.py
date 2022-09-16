import pprint
import click # type: ignore

import zetta_utils as zu

# For now, CLI requires all modules to be installed
# If the need arises, the installed modules can be specified
# through a config file.
zu.load_all_modules()


@click.group()
def cli():
    pass  # pragma: no cover


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
    """Perform ``zu.builder.build`` action on file contents."""
    spec = zu.cue.load(path)
    result = zu.builder.build(spec)
    pprint.pprint(result)
    if pdb:
        breakpoint()  # pylint: disable=forgotten-debug-statement # pragma: no cover


@click.command()
def show_registry():
    """Display builder registry."""
    pprint.pprint(zu.builder.REGISTRY)


cli.add_command(run)
cli.add_command(show_registry)


if __name__ == "__main__":
    cli()  # pragma: no cover

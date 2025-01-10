import os
import time
from collections import namedtuple
from datetime import datetime
from typing import Final

import click
from rich import print as rich_print
from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint
from rich.table import Table

from . import RUN_DB, RUN_INFO_BUCKET, RunInfo, get_latest_checkpoint

COLUMNS: Final = namedtuple(
    "COLUMNS", ["zetta_user", "state", "timestamp", "heartbeat", "run_id", "duration_s"]
)


def _print_infos(infos: list) -> Table:
    table = Table()
    table.add_column(COLUMNS._fields[0], justify="right", style="cyan", no_wrap=True)
    table.add_column(COLUMNS._fields[1], style="magenta", no_wrap=True)
    table.add_column(COLUMNS._fields[2], style="green")
    table.add_column(COLUMNS._fields[3], style="green")
    table.add_column(COLUMNS._fields[4], justify="left", style="blue", no_wrap=True)
    table.add_column(COLUMNS._fields[5], justify="left", style="green")

    for info in infos[::-1]:
        timestamp = info.get(COLUMNS._fields[2], 0)
        heartbeat = info.get(COLUMNS._fields[3], timestamp)
        info[COLUMNS._fields[2]] = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        info[COLUMNS._fields[3]] = datetime.fromtimestamp(heartbeat).strftime("%Y-%m-%d %H:%M:%S")
        info[COLUMNS._fields[5]] = int(heartbeat - timestamp)
        row = [str(info.get(key, None)) for key in COLUMNS._fields]
        table.add_row(*row)
    return table


@click.group()
def run_info_cli():
    ...


@run_info_cli.command()
@click.argument("run_ids", type=str, nargs=-1)
def run_info(run_ids: list[str]):
    """
    Display information about `run_id [[run_id] ...]`
    """
    info_path = os.environ.get("RUN_INFO_BUCKET", RUN_INFO_BUCKET)
    infos = RUN_DB[(run_ids, (x.value for x in RunInfo))]
    for run_id, info in zip(run_ids, infos):
        rich_print(Panel(run_id))

        zetta_user = info[COLUMNS._fields[0]]
        timestamp = info.get(COLUMNS._fields[2], 0)
        heartbeat = info.get(COLUMNS._fields[3], timestamp)
        info[COLUMNS._fields[2]] = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        info[COLUMNS._fields[3]] = datetime.fromtimestamp(heartbeat).strftime("%Y-%m-%d %H:%M:%S")
        info[COLUMNS._fields[5]] = int(heartbeat - timestamp)

        info_path_user = os.path.join(info_path, zetta_user)
        info["spec_path"] = os.path.join(info_path_user, f"{run_id}.cue")
        info["last_checkpoint"] = get_latest_checkpoint(run_id, zetta_user=zetta_user)
        pprint(info, expand_all=True)


@run_info_cli.command()
@click.argument("user", type=str, required=False)
@click.option("-d", "--days", type=int, default=7, help="Limit by number of days, defaults to 7.")
def run_list(user: str, days: int):
    """
    Display list of runs with basic information.

    Sorted by `timestamp` (desc). Can filter by `user`.
    """
    _filter: dict[str, list] = {f">{COLUMNS._fields[2]}": [time.time() - 24 * 3600 * days]}
    if user:
        _filter[COLUMNS._fields[0]] = [user]
    result = RUN_DB.query(_filter, union=False)
    for k, v in result.items():
        v["run_id"] = k
    table = _print_infos([*result.values()])
    console = Console()
    console.print(table)

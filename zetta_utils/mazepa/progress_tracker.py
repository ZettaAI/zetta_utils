from __future__ import annotations

import contextlib
import os
import signal
import sys
import time
from typing import Generator, Protocol

import fsspec
import rich
from rich import progress
from rich.console import Console

from zetta_utils.common import custom_signal_handler_ctx, get_user_confirmation
from zetta_utils.log import get_logger

from .execution_state import ProgressReport

logger = get_logger("zetta_utils")


def get_confirm_sigint_fn(progress_bar: progress.Progress):  # pragma: no cover
    def handler(_, __):
        progress_bar.stop()
        interrupt = False
        try:
            confirmed = get_user_confirmation(
                prompt="Confirm sending KeyboardInterrupt? ", timeout=7
            )
            if confirmed is None:
                rich.print("\nNo input for 7 seconds. Resuming...")
            elif not confirmed:
                rich.print("Resuming...")
            else:
                interrupt = True
        except KeyboardInterrupt:
            interrupt = True

        if interrupt:
            raise KeyboardInterrupt
        progress_bar.start()

    return handler


def graceful_shutdown(_, __):  # pragma: no cover
    logger.info("Detected signal")
    raise RuntimeError("Cancelled")


class ProgressUpdateFN(Protocol):
    def __call__(self, progress_reports: dict[str, ProgressReport]) -> None:
        ...


@contextlib.contextmanager
def progress_ctx_mngr(
    expected_total_counts: dict[str, int],
    write_progress_to_path: str | None = None,
    write_progress_interval_sec: int = 5,
    require_interrupt_confirm: bool = True,
) -> Generator[ProgressUpdateFN, None, None]:  # pragma: no cover
    progress_bar = progress.Progress(
        progress.SpinnerColumn(),
        progress.TextColumn("[progress.description]{task.description}"),
        progress.BarColumn(),
        progress.TaskProgressColumn(),
        progress.TextColumn("[progress.description] {task.completed}/{task.total}"),
        progress.TimeElapsedColumn(),
        progress.TimeRemainingColumn(),
        transient=True,
        speed_estimate_period=90,
        # refresh_per_second=1,
    )

    def custom_debugger_hook():
        # reading the value of the environment variable
        val = os.environ.get("PYTHONBREAKPOINT")
        # if the value has been set to 0, skip all breakpoints
        if val == "0":
            return None
        # else if the value is an empty string, invoke the default pdb debugger
        elif val is not None and not val.endswith(".set_trace"):
            raise Exception(  # pylint: disable=broad-exception-raised
                "Custom debuggers are not allowed when `rich.progress is in use.`"
            )

        progress_bar.stop()
        try:
            import ipdb  # pylint: disable=import-outside-toplevel

            return ipdb.set_trace(sys._getframe().f_back)  # pylint: disable=protected-access
        except ImportError:
            import pdb  # pylint: disable=import-outside-toplevel

            return pdb.Pdb().set_trace(sys._getframe().f_back)  # pylint: disable=protected-access

    sys.breakpointhook = custom_debugger_hook
    last_progress_writeout_ts = 0.0

    with progress_bar as progress_bar:
        if require_interrupt_confirm:
            handler_ctx = custom_signal_handler_ctx(
                get_confirm_sigint_fn(progress_bar), signal.SIGINT
            )
        else:
            handler_ctx = custom_signal_handler_ctx(graceful_shutdown, signal.SIGTERM)
        with handler_ctx:
            submission_tracker_ids = {
                k: progress_bar.add_task(
                    f"[cyan]Submission {k}",
                    total=v,
                    start=False,
                    spinner_color="cyan",
                    auto_refresh=False,
                )
                for k, v in expected_total_counts.items()
            }

            execution_tracker_ids: dict[str, progress.TaskID] = {}

            def write_progress_file():
                temp_console = Console(record=True, width=80)
                for line in progress_bar.get_renderables():
                    temp_console.print(line)
                progress_html = temp_console.export_html(inline_styles=True)

                with fsspec.open(write_progress_to_path, "w") as f:
                    f.write(progress_html)

            def update_fn(progress_reports: dict[str, ProgressReport]) -> None:
                nonlocal last_progress_writeout_ts
                if not hasattr(sys, "gettrace") or sys.gettrace() is None:
                    progress_bar.start()

                for k, v in progress_reports.items():
                    if k in submission_tracker_ids:
                        progress_bar.start_task(submission_tracker_ids[k])
                        progress_bar.update(submission_tracker_ids[k], completed=v.submitted_count)

                    if k not in execution_tracker_ids:
                        execution_tracker_ids[k] = progress_bar.add_task(
                            f"[green]Completion {k}", start=False, spinner_color="green"
                        )

                    progress_bar.update(execution_tracker_ids[k], total=v.submitted_count)

                    if v.completed_count != 0:
                        progress_bar.start_task(execution_tracker_ids[k])
                        progress_bar.update(execution_tracker_ids[k], completed=v.completed_count)

                progress_bar.refresh()
                if (write_progress_to_path is not None) and (
                    time.time() - last_progress_writeout_ts > write_progress_interval_sec
                ):
                    write_progress_file()

                    last_progress_writeout_ts = time.time()

            yield update_fn
            if write_progress_to_path is not None:
                write_progress_file()
            try:
                progress_bar.stop()
            except IndexError:
                pass
